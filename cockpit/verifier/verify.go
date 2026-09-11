// Package verifier independently checks OpenManus cockpit trace exports.
package verifier

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

const chainDomain = "openmanus-cockpit-event-v1"

type Export struct {
	SchemaVersion int     `json:"schema_version"`
	Task          Task    `json:"task"`
	Events        []Event `json:"events"`
}

type Task struct {
	ID                string `json:"id"`
	Status            string `json:"status"`
	HeadHash          string `json:"head_hash"`
	EvidenceCount     int    `json:"evidence_count"`
	UnresolvedEffects int    `json:"unresolved_effects"`
}

type Event struct {
	TaskID           string          `json:"task_id"`
	Position         int             `json:"position"`
	EventID          string          `json:"event_id"`
	Kind             string          `json:"kind"`
	StateBefore      string          `json:"state_before"`
	StateAfter       string          `json:"state_after"`
	Payload          json.RawMessage `json:"payload"`
	PayloadCanonical string          `json:"payload_canonical"`
	PayloadSHA256    string          `json:"payload_sha256"`
	PrevHash         string          `json:"prev_hash"`
	EventHash        string          `json:"event_hash"`
	CreatedAt        string          `json:"created_at"`
}

type Result struct {
	Valid           bool            `json:"valid"`
	Errors          []string        `json:"errors"`
	EventCount      int             `json:"event_count"`
	HeadHash        string          `json:"head_hash"`
	RuntimeReceipts []RuntimeResult `json:"runtime_receipts,omitempty"`
}

func hashText(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

func hashEvent(event Event) string {
	preimage := strings.Join([]string{
		chainDomain,
		event.TaskID,
		strconv.Itoa(event.Position),
		event.EventID,
		event.Kind,
		event.StateBefore,
		event.StateAfter,
		event.PayloadSHA256,
		event.CreatedAt,
		event.PrevHash,
	}, "\n")
	return hashText(preimage)
}

// VerifyJSON checks payload integrity, event order, state transitions, terminal
// gates, and the per-task SHA-256 chain. SHA-256 is an integrity check here,
// not an author signature.
func VerifyJSON(data []byte) Result {
	result := Result{Errors: []string{}}
	var document Export
	if err := json.Unmarshal(data, &document); err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("invalid JSON: %v", err))
		return result
	}
	result.EventCount = len(document.Events)
	if document.SchemaVersion != 1 {
		result.Errors = append(result.Errors, "schema_version must be 1")
	}
	if document.Task.ID == "" {
		result.Errors = append(result.Errors, "task id must not be empty")
	}
	if len(document.Events) == 0 {
		result.Errors = append(result.Errors, "trace must contain at least one event")
	}

	state := ""
	prevHash := ""
	evidenceCount := 0
	unresolvedEffects := 0
	for index, event := range document.Events {
		position := index + 1
		label := fmt.Sprintf("event %d", position)
		if event.Position != position {
			result.Errors = append(result.Errors, label+" position mismatch")
		}
		if event.TaskID != document.Task.ID {
			result.Errors = append(result.Errors, label+" task_id mismatch")
		}
		if event.StateBefore != state {
			result.Errors = append(result.Errors, label+" state_before mismatch")
		}
		if event.PrevHash != prevHash {
			result.Errors = append(result.Errors, label+" prev_hash mismatch")
		}
		var canonicalPayload any
		var payloadView any
		if err := json.Unmarshal([]byte(event.PayloadCanonical), &canonicalPayload); err != nil {
			result.Errors = append(result.Errors, label+" payload_canonical is invalid JSON")
		}
		if err := json.Unmarshal(event.Payload, &payloadView); err != nil {
			result.Errors = append(result.Errors, label+" payload view is invalid JSON")
		} else if !reflect.DeepEqual(payloadView, canonicalPayload) {
			result.Errors = append(result.Errors, label+" payload view does not match canonical payload")
		}
		if hashText(event.PayloadCanonical) != event.PayloadSHA256 {
			result.Errors = append(result.Errors, label+" payload_sha256 mismatch")
		}

		switch event.Kind {
		case "task.created":
			if state != "" || event.StateAfter != "created" {
				result.Errors = append(result.Errors, label+" invalid task.created transition")
			}
		case "task.accepted":
			if state != "created" || event.StateAfter != "accepted" {
				result.Errors = append(result.Errors, label+" invalid task.accepted transition")
			}
		case "run.started":
			if state != "accepted" || event.StateAfter != "running" {
				result.Errors = append(result.Errors, label+" invalid run.started transition")
			}
		case "evidence.recorded":
			if state != "running" || event.StateAfter != state {
				result.Errors = append(result.Errors, label+" evidence outside running state")
			}
			evidenceCount++
			var envelope struct {
				EvidenceType string          `json:"evidence_type"`
				Value        json.RawMessage `json:"value"`
			}
			if err := json.Unmarshal(event.Payload, &envelope); err == nil && envelope.EvidenceType == "formal.runtime.conformance.v1" {
				runtimeResult, receiptErrors := verifyFormalReceipt(envelope.Value)
				result.RuntimeReceipts = append(result.RuntimeReceipts, runtimeResult)
				for _, receiptError := range receiptErrors {
					result.Errors = append(result.Errors, label+" "+receiptError)
				}
			}
		case "effect.unknown":
			if (state != "accepted" && state != "running") || event.StateAfter != state {
				result.Errors = append(result.Errors, label+" invalid effect.unknown state")
			}
			unresolvedEffects++
		case "effect.resolved":
			if (state != "accepted" && state != "running") || event.StateAfter != state {
				result.Errors = append(result.Errors, label+" invalid effect.resolved state")
			}
			unresolvedEffects--
			if unresolvedEffects < 0 {
				result.Errors = append(result.Errors, label+" resolved more effects than were opened")
			}
		case "run.completed":
			if state != "running" || event.StateAfter != "completed" {
				result.Errors = append(result.Errors, label+" invalid run.completed transition")
			}
			if evidenceCount == 0 {
				result.Errors = append(result.Errors, label+" completed without evidence")
			}
			if unresolvedEffects != 0 {
				result.Errors = append(result.Errors, label+" completed with unresolved effects")
			}
		case "run.failed":
			if (state != "created" && state != "accepted" && state != "running") || event.StateAfter != "failed" {
				result.Errors = append(result.Errors, label+" invalid run.failed transition")
			}
		default:
			result.Errors = append(result.Errors, label+" unknown event kind: "+event.Kind)
		}

		calculatedHash := hashEvent(event)
		if event.EventHash != calculatedHash {
			result.Errors = append(result.Errors, label+" event_hash mismatch")
		}
		prevHash = calculatedHash
		state = event.StateAfter
		if (state == "completed" || state == "failed") && position != len(document.Events) {
			result.Errors = append(result.Errors, label+" terminal event is not last")
		}
	}

	if document.Task.Status != state {
		result.Errors = append(result.Errors, "task status does not match trace")
	}
	if document.Task.HeadHash != prevHash {
		result.Errors = append(result.Errors, "task head_hash does not match trace")
	}
	if document.Task.EvidenceCount != evidenceCount {
		result.Errors = append(result.Errors, "task evidence_count does not match trace")
	}
	if document.Task.UnresolvedEffects != unresolvedEffects {
		result.Errors = append(result.Errors, "task unresolved_effects does not match trace")
	}
	if document.Task.Status == "completed" {
		for index, runtimeResult := range result.RuntimeReceipts {
			if !runtimeResult.Valid {
				result.Errors = append(
					result.Errors,
					fmt.Sprintf("runtime receipt %d is not conformant", index+1),
				)
			}
		}
	}
	result.HeadHash = prevHash
	result.Valid = len(result.Errors) == 0
	return result
}
