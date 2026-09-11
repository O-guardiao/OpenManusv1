package verifier

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const (
	runtimeChainDomain = "openmanus-oak-runtime-event-v1"
	runtimeContract    = "oak-openmanus-runtime-v1"
)

var budgetDimensions = []string{
	"steps",
	"model_calls",
	"tool_calls",
	"retries",
	"wall_time_ms",
	"tokens",
	"context_bytes",
	"external_effects",
}

var unresolvedEffectStatuses = map[string]bool{
	"dispatched": true,
	"unknown":    true,
	"partial":    true,
}

var resolvedEffectStatuses = map[string]bool{
	"succeeded":            true,
	"failed_before_effect": true,
	"verified_applied":     true,
	"verified_absent":      true,
	"compensated":          true,
	"manual_repair":        true,
}

// RuntimeResult is the independent Go/WASM projection of one Python OaK trace.
type RuntimeResult struct {
	Valid             bool              `json:"valid"`
	Errors            []string          `json:"errors"`
	EventCount        int               `json:"event_count"`
	HeadHash          string            `json:"head_hash"`
	EvidenceCount     int               `json:"evidence_count"`
	UnresolvedEffects int               `json:"unresolved_effects"`
	BudgetExhausted   bool              `json:"budget_exhausted"`
	TerminalStatus    string            `json:"terminal_status"`
	BudgetUsage       map[string]int    `json:"budget_usage"`
	ProviderStates    map[string]string `json:"provider_states"`
	requestSHA256     string
	resultSHA256      string
}

type formalRuntimeReceipt struct {
	RequestSHA256          string           `json:"request_sha256"`
	ResultSHA256           string           `json:"result_sha256"`
	OakTraceSHA256         string           `json:"oak_trace_sha256"`
	OakTrace               []map[string]any `json:"oak_trace"`
	OakEvidenceCount       int              `json:"oak_evidence_count"`
	TerminalStatus         string           `json:"terminal_status"`
	RuntimeContractVersion string           `json:"runtime_contract_version"`
	FormalConformance      RuntimeResult    `json:"formal_conformance"`
	EvidenceCountAgrees    bool             `json:"evidence_count_agrees"`
	TerminalStatusAgrees   bool             `json:"terminal_status_agrees"`
	RequestDigestAgrees    bool             `json:"request_digest_agrees"`
	ResultDigestAgrees     bool             `json:"result_digest_agrees"`
}

func canonicalJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), nil
}

func runtimeEventHash(event map[string]any) (string, error) {
	payload := make(map[string]any, len(event)-1)
	for key, value := range event {
		if key != "event_hash" {
			payload[key] = value
		}
	}
	canonical, err := canonicalJSON(payload)
	if err != nil {
		return "", err
	}
	preimage := append([]byte(runtimeChainDomain+"\n"), canonical...)
	digest := sha256.Sum256(preimage)
	return hex.EncodeToString(digest[:]), nil
}

func stringField(event map[string]any, key string) string {
	value, _ := event[key].(string)
	return value
}

func intField(event map[string]any, key string) (int, bool) {
	switch value := event[key].(type) {
	case float64:
		integer := int(value)
		return integer, value == float64(integer)
	case int:
		return value, true
	default:
		return 0, false
	}
}

func boolField(event map[string]any, key string) (bool, bool) {
	value, ok := event[key].(bool)
	return value, ok
}

func stringSliceField(event map[string]any, key string) []string {
	raw, _ := event[key].([]any)
	values := make([]string, 0, len(raw))
	for _, item := range raw {
		if value, ok := item.(string); ok {
			values = append(values, value)
		}
	}
	return values
}

func verifyRuntimeTrace(events []map[string]any) RuntimeResult {
	result := RuntimeResult{
		Errors:         []string{},
		EventCount:     len(events),
		BudgetUsage:    map[string]int{},
		ProviderStates: map[string]string{},
	}
	for _, dimension := range budgetDimensions {
		result.BudgetUsage[dimension] = 0
	}
	if len(events) == 0 {
		result.Errors = append(result.Errors, "trace must not be empty")
	}

	previousHash := ""
	sessionID := ""
	taskStarted := 0
	taskFinished := 0
	kernelHash := ""
	declaredLimits := map[string]any{}
	allowedGates := map[string]bool{}
	effects := map[string]string{}
	idempotencyKeys := map[string]bool{}
	seenEventIDs := map[string]bool{}
	var terminal map[string]any

	for index, event := range events {
		position := index + 1
		label := fmt.Sprintf("runtime event %d", position)
		eventType := stringField(event, "event_type")
		if taskFinished > 0 {
			result.Errors = append(result.Errors, label+" appears after terminal event")
		}
		if version, ok := intField(event, "schema_version"); !ok || version != 1 {
			result.Errors = append(result.Errors, label+" schema_version mismatch")
		}
		if stringField(event, "contract_version") != runtimeContract {
			result.Errors = append(result.Errors, label+" contract_version mismatch")
		}
		eventID := stringField(event, "event_id")
		if eventID == "" || seenEventIDs[eventID] {
			result.Errors = append(result.Errors, label+" duplicate or empty event identity")
		}
		seenEventIDs[eventID] = true
		if position == 1 && eventType != "task_started" {
			result.Errors = append(result.Errors, label+" trace must start with task_started")
		} else if position > 1 && taskStarted == 0 {
			result.Errors = append(result.Errors, label+" appears before task_started")
		}
		if actual, ok := intField(event, "position"); !ok || actual != position {
			result.Errors = append(result.Errors, label+" position mismatch")
		}
		if stringField(event, "prev_hash") != previousHash {
			result.Errors = append(result.Errors, label+" prev_hash mismatch")
		}
		currentSession := stringField(event, "session_id")
		if position == 1 {
			sessionID = currentSession
		} else if currentSession != sessionID {
			result.Errors = append(result.Errors, label+" session_id mismatch")
		}
		calculated, err := runtimeEventHash(event)
		if err != nil {
			result.Errors = append(result.Errors, label+" cannot be canonicalized")
		} else {
			if stringField(event, "event_hash") != calculated {
				result.Errors = append(result.Errors, label+" event_hash mismatch")
			}
			previousHash = calculated
		}

		switch eventType {
		case "task_started":
			taskStarted++
			if taskStarted > 1 || taskFinished > 0 {
				result.Errors = append(result.Errors, label+" invalid task start ordering")
			}
			kernelHash = stringField(event, "kernel_sha256")
			if kernelHash == "" {
				result.Errors = append(result.Errors, label+" missing frozen kernel hash")
			}
			result.requestSHA256 = stringField(event, "request_sha256")
			if limits, ok := event["budget_limits"].(map[string]any); ok {
				declaredLimits = limits
				for _, dimension := range budgetDimensions {
					limit := limits[dimension]
					if limit == nil {
						continue
					}
					value, valid := intField(limits, dimension)
					if !valid || value < 0 {
						result.Errors = append(result.Errors, label+" invalid budget limit for "+dimension)
					}
				}
			} else {
				result.Errors = append(result.Errors, label+" missing budget limits")
			}
		case "task_finished":
			taskFinished++
			terminal = event
			result.TerminalStatus = stringField(event, "status")
			if result.TerminalStatus == "" {
				result.Errors = append(result.Errors, label+" missing terminal status")
			}
			if taskStarted != 1 || taskFinished > 1 {
				result.Errors = append(result.Errors, label+" invalid task finish ordering")
			}
			if stringField(event, "kernel_sha256") != kernelHash {
				result.Errors = append(result.Errors, label+" kernel changed during task")
			}
		case "budget_consumed":
			dimension := stringField(event, "dimension")
			current, known := result.BudgetUsage[dimension]
			delta, deltaOK := intField(event, "delta")
			used, usedOK := intField(event, "used")
			if !known || !deltaOK || delta < 0 || !usedOK || used < 0 {
				result.Errors = append(result.Errors, label+" invalid budget event")
				break
			}
			if event["limit"] != nil {
				limit, limitOK := intField(event, "limit")
				if !limitOK || limit < 0 {
					result.Errors = append(result.Errors, label+" invalid budget limit")
					break
				}
			}
			current += delta
			result.BudgetUsage[dimension] = current
			if used != current {
				result.Errors = append(result.Errors, label+" non-monotonic budget usage")
			}
			if limit, exists := event["limit"]; exists && declaredLimits[dimension] != limit {
				result.Errors = append(result.Errors, label+" budget limit differs from task contract")
			}
			if limit, ok := intField(event, "limit"); ok && current > limit {
				result.Errors = append(result.Errors, label+" budget exceeded after admission")
			}
		case "budget_rejected":
			result.BudgetExhausted = true
			if _, known := result.BudgetUsage[stringField(event, "dimension")]; !known {
				result.Errors = append(result.Errors, label+" unknown rejected budget dimension")
			}
		case "source_result", "internal_result":
			result.EvidenceCount++
		case "final_evidence":
			result.EvidenceCount++
			result.resultSHA256 = stringField(event, "content_sha256")
		case "sink_decision":
			allowed, _ := boolField(event, "allowed")
			if allowed && stringField(event, "outcome") == "allow" {
				gateID := stringField(event, "event_id")
				if gateID == "" {
					result.Errors = append(result.Errors, label+" allowed sink gate has no identity")
				}
				if len(stringSliceField(event, "taint_ids")) > 0 && stringField(event, "approval_key") == "" {
					result.Errors = append(result.Errors, label+" tainted sink lacks exact approval key")
				}
				allowedGates[gateID] = true
			}
		case "effect_intent":
			operationID := stringField(event, "operation_id")
			if operationID == "" || effects[operationID] != "" {
				result.Errors = append(result.Errors, label+" duplicate or empty effect operation")
			}
			if !allowedGates[stringField(event, "gate_event_id")] {
				result.Errors = append(result.Errors, label+" effect intent lacks prior allowed gate")
			}
			key := stringField(event, "idempotency_key")
			if key == "" || idempotencyKeys[key] {
				result.Errors = append(result.Errors, label+" missing or duplicate idempotency key")
			}
			idempotencyKeys[key] = true
			effects[operationID] = "intent_logged"
		case "effect_dispatched":
			operationID := stringField(event, "operation_id")
			if effects[operationID] != "intent_logged" {
				result.Errors = append(result.Errors, label+" dispatch without durable intent")
			}
			effects[operationID] = "dispatched"
		case "effect_observed":
			operationID := stringField(event, "operation_id")
			current := effects[operationID]
			if current != "intent_logged" && !unresolvedEffectStatuses[current] {
				result.Errors = append(result.Errors, label+" observation without live operation")
			}
			status := stringField(event, "status")
			if !unresolvedEffectStatuses[status] && !resolvedEffectStatuses[status] {
				result.Errors = append(result.Errors, label+" invalid effect status")
			}
			effects[operationID] = status
		case "provider_state":
			providerID := stringField(event, "provider_sha256")
			before := result.ProviderStates[providerID]
			if before == "" {
				before = "absent"
			}
			after := stringField(event, "state_after")
			allowed := map[string]map[string]bool{
				"absent":         {"opening": true},
				"opening":        {"active": true, "absent": true, "failed_cleanup": true},
				"active":         {"draining": true},
				"draining":       {"absent": true, "failed_cleanup": true},
				"failed_cleanup": {"draining": true},
			}
			if providerID == "" || stringField(event, "state_before") != before {
				result.Errors = append(result.Errors, label+" provider pre-state mismatch")
			}
			if !allowed[before][after] {
				result.Errors = append(result.Errors, label+" invalid provider transition")
			}
			if count, ok := intField(event, "published_tool_count"); !ok || count < 0 || (after != "active" && count != 0) {
				result.Errors = append(result.Errors, label+" invalid provider publication count")
			}
			result.ProviderStates[providerID] = after
		}
	}

	for _, status := range effects {
		if unresolvedEffectStatuses[status] {
			result.UnresolvedEffects++
		}
	}
	if taskStarted != 1 || taskFinished != 1 {
		result.Errors = append(result.Errors, "trace must contain exactly one task start and finish")
	}
	if terminal != nil {
		if value, ok := intField(terminal, "evidence_count"); !ok || value != result.EvidenceCount {
			result.Errors = append(result.Errors, "terminal evidence count does not match trace")
		}
		if value, ok := intField(terminal, "unresolved_effects"); !ok || value != result.UnresolvedEffects {
			result.Errors = append(result.Errors, "terminal unresolved-effect count does not match trace")
		}
		if value, ok := boolField(terminal, "budget_exhausted"); !ok || value != result.BudgetExhausted {
			result.Errors = append(result.Errors, "terminal budget flag does not match trace")
		}
		if usage, ok := terminal["budget_usage"].(map[string]any); ok {
			for _, dimension := range budgetDimensions {
				value, valueOK := intField(usage, dimension)
				if !valueOK || value != result.BudgetUsage[dimension] {
					result.Errors = append(result.Errors, "terminal budget usage does not match trace")
					break
				}
			}
		} else {
			result.Errors = append(result.Errors, "terminal budget usage does not match trace")
		}
		terminalProviders, ok := terminal["provider_states"].(map[string]any)
		if !ok || len(terminalProviders) != len(result.ProviderStates) {
			result.Errors = append(result.Errors, "terminal provider states do not match trace")
		} else {
			for providerID, state := range result.ProviderStates {
				if terminalProviders[providerID] != state {
					result.Errors = append(result.Errors, "terminal provider states do not match trace")
					break
				}
			}
		}
	}
	if result.TerminalStatus == "completed" && result.EvidenceCount == 0 {
		result.Errors = append(result.Errors, "completed task has no evidence")
	}
	if result.TerminalStatus == "completed" && result.UnresolvedEffects > 0 {
		result.Errors = append(result.Errors, "completed task has unresolved effects")
	}
	if result.TerminalStatus == "completed" && result.BudgetExhausted {
		result.Errors = append(result.Errors, "completed task exceeded a global budget")
	}
	if result.TerminalStatus == "completed" {
		for _, state := range result.ProviderStates {
			if state != "absent" {
				result.Errors = append(result.Errors, "completed task has a live or failed provider")
				break
			}
		}
	}
	result.HeadHash = previousHash
	result.Valid = len(result.Errors) == 0
	return result
}

func verifyFormalReceipt(raw json.RawMessage) (RuntimeResult, []string) {
	var receipt formalRuntimeReceipt
	if err := json.Unmarshal(raw, &receipt); err != nil {
		return RuntimeResult{Errors: []string{"invalid formal runtime receipt"}}, []string{"formal runtime receipt is invalid JSON"}
	}
	recomputed := verifyRuntimeTrace(receipt.OakTrace)
	errors := []string{}
	canonical, err := canonicalJSON(receipt.OakTrace)
	if err != nil || hashText(string(canonical)) != receipt.OakTraceSHA256 {
		errors = append(errors, "formal receipt oak_trace_sha256 mismatch")
	}
	if receipt.RuntimeContractVersion != runtimeContract {
		errors = append(errors, "formal receipt runtime contract mismatch")
	}
	if !reflectRuntimeResult(receipt.FormalConformance, recomputed) {
		errors = append(errors, "formal receipt conformance claim mismatch")
	}
	if receipt.OakEvidenceCount != recomputed.EvidenceCount || !receipt.EvidenceCountAgrees {
		errors = append(errors, "formal receipt evidence count mismatch")
	}
	if receipt.TerminalStatus != recomputed.TerminalStatus || !receipt.TerminalStatusAgrees {
		errors = append(errors, "formal receipt terminal status mismatch")
	}
	if receipt.RequestSHA256 != recomputed.requestSHA256 || !receipt.RequestDigestAgrees {
		errors = append(errors, "formal receipt request digest mismatch")
	}
	if receipt.ResultSHA256 != recomputed.resultSHA256 || !receipt.ResultDigestAgrees {
		errors = append(errors, "formal receipt result digest mismatch")
	}
	return recomputed, errors
}

func reflectRuntimeResult(left, right RuntimeResult) bool {
	leftJSON, leftErr := canonicalJSON(left)
	rightJSON, rightErr := canonicalJSON(right)
	return leftErr == nil && rightErr == nil && bytes.Equal(leftJSON, rightJSON)
}
