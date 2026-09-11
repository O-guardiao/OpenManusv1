package verifier

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"
)

func TestVerifyJSONAcceptsValidTraceAndRejectsPayloadMutation(t *testing.T) {
	document := fixture(t)
	encoded, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}

	result := VerifyJSON(encoded)
	if !result.Valid {
		t.Fatalf("expected valid trace, got %v", result.Errors)
	}

	document.Events[3].PayloadCanonical = `{"evidence_type":"repository.snapshot","value":{"files":999}}`
	mutated, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	result = VerifyJSON(mutated)
	if result.Valid {
		t.Fatal("payload mutation was accepted")
	}

	document = fixture(t)
	document.Events[3].Payload = json.RawMessage(`{"forged":true}`)
	mutated, err = json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if VerifyJSON(mutated).Valid {
		t.Fatal("payload view mutation was accepted")
	}
}

func TestVerifyJSONIndependentlyChecksFormalRuntimeReceipt(t *testing.T) {
	document := fixture(t)
	receipt := runtimeReceiptFixture(t)
	setFormalReceipt(t, &document, receipt)
	encoded, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}

	result := VerifyJSON(encoded)
	if !result.Valid {
		t.Fatalf("expected formal receipt to be valid, got %v", result.Errors)
	}
	if len(result.RuntimeReceipts) != 1 || !result.RuntimeReceipts[0].Valid {
		t.Fatalf("runtime receipt was not independently accepted: %+v", result.RuntimeReceipts)
	}

	receipt.OakTrace[1]["content_sha256"] = "tampered"
	setFormalReceipt(t, &document, receipt)
	encoded, err = json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	result = VerifyJSON(encoded)
	if result.Valid {
		t.Fatal("nested runtime trace mutation was accepted")
	}
}

func runtimeReceiptFixture(t *testing.T) formalRuntimeReceipt {
	t.Helper()
	limits := map[string]any{}
	usage := map[string]any{}
	for _, dimension := range budgetDimensions {
		limits[dimension] = float64(10)
		usage[dimension] = float64(0)
	}
	events := []map[string]any{
		{
			"schema_version":   float64(1),
			"contract_version": runtimeContract,
			"event_id":         "runtime-1",
			"event_type":       "task_started",
			"session_id":       "runtime-session",
			"occurred_at":      "2026-09-03T12:00:00Z",
			"position":         float64(1),
			"prev_hash":        "",
			"task_id":          "task-runtime",
			"request_sha256":   "request-digest",
			"kernel_sha256":    "kernel-digest",
			"policy_version":   "policy-v2",
			"budget_limits":    limits,
		},
		{
			"schema_version":   float64(1),
			"contract_version": runtimeContract,
			"event_id":         "runtime-2",
			"event_type":       "final_evidence",
			"session_id":       "runtime-session",
			"occurred_at":      "2026-09-03T12:00:01Z",
			"position":         float64(2),
			"prev_hash":        "",
			"content_sha256":   "result-digest",
			"characters":       float64(4),
		},
		{
			"schema_version":     float64(1),
			"contract_version":   runtimeContract,
			"event_id":           "runtime-3",
			"event_type":         "task_finished",
			"session_id":         "runtime-session",
			"occurred_at":        "2026-09-03T12:00:02Z",
			"position":           float64(3),
			"prev_hash":          "",
			"task_id":            "task-runtime",
			"status":             "completed",
			"requested_status":   "completed",
			"evidence_count":     float64(1),
			"active_taint_count": float64(0),
			"unresolved_effects": float64(0),
			"budget_exhausted":   false,
			"budget_usage":       usage,
			"provider_states":    map[string]any{},
			"kernel_sha256":      "kernel-digest",
		},
	}
	previous := ""
	for _, event := range events {
		event["prev_hash"] = previous
		hash, err := runtimeEventHash(event)
		if err != nil {
			t.Fatal(err)
		}
		event["event_hash"] = hash
		previous = hash
	}
	report := verifyRuntimeTrace(events)
	canonical, err := canonicalJSON(events)
	if err != nil {
		t.Fatal(err)
	}
	return formalRuntimeReceipt{
		RequestSHA256:          "request-digest",
		ResultSHA256:           "result-digest",
		OakTraceSHA256:         hashText(string(canonical)),
		OakTrace:               events,
		OakEvidenceCount:       1,
		TerminalStatus:         "completed",
		RuntimeContractVersion: runtimeContract,
		FormalConformance:      report,
		EvidenceCountAgrees:    true,
		TerminalStatusAgrees:   true,
		RequestDigestAgrees:    true,
		ResultDigestAgrees:     true,
	}
}

func setFormalReceipt(t *testing.T, document *Export, receipt formalRuntimeReceipt) {
	t.Helper()
	payload, err := canonicalJSON(map[string]any{
		"evidence_type": "formal.runtime.conformance.v1",
		"value":         receipt,
	})
	if err != nil {
		t.Fatal(err)
	}
	document.Events[3].Payload = json.RawMessage(payload)
	document.Events[3].PayloadCanonical = string(payload)
	document.Events[3].PayloadSHA256 = hashText(string(payload))
	previous := document.Events[2].EventHash
	for index := 3; index < len(document.Events); index++ {
		document.Events[index].PrevHash = previous
		document.Events[index].EventHash = hashEvent(document.Events[index])
		previous = document.Events[index].EventHash
	}
	document.Task.HeadHash = previous
}

func fixture(t *testing.T) Export {
	t.Helper()
	taskID := "11111111-1111-1111-1111-111111111111"
	events := []Event{}
	prev := ""
	states := []struct {
		kind, before, after, payload string
	}{
		{"task.created", "", "created", `{"request_key":"fixture"}`},
		{"task.accepted", "created", "accepted", `{}`},
		{"run.started", "accepted", "running", `{}`},
		{"evidence.recorded", "running", "running", `{"evidence_type":"repository.snapshot","value":{"files":1}}`},
		{"run.completed", "running", "completed", `{"decision":"clean_room_only"}`},
	}
	for index, item := range states {
		payloadDigest := sha256.Sum256([]byte(item.payload))
		event := Event{
			TaskID:           taskID,
			Position:         index + 1,
			EventID:          string(rune('a' + index)),
			Kind:             item.kind,
			StateBefore:      item.before,
			StateAfter:       item.after,
			Payload:          json.RawMessage(item.payload),
			PayloadCanonical: item.payload,
			PayloadSHA256:    hex.EncodeToString(payloadDigest[:]),
			PrevHash:         prev,
			CreatedAt:        "2026-09-03T12:00:00.000Z",
		}
		event.EventHash = hashEvent(event)
		prev = event.EventHash
		events = append(events, event)
	}
	return Export{
		SchemaVersion: 1,
		Task: Task{
			ID:                taskID,
			Status:            "completed",
			HeadHash:          prev,
			EvidenceCount:     1,
			UnresolvedEffects: 0,
		},
		Events: events,
	}
}
