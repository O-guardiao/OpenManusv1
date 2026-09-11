//go:build js && wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/O-guardiao/OpenManusv1/cockpit/verifier"
)

func main() {
	verify := js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) != 1 {
			encoded, _ := json.Marshal(verifier.Result{
				Valid:  false,
				Errors: []string{"verify expects one JSON string"},
			})
			return string(encoded)
		}
		result := verifier.VerifyJSON([]byte(args[0].String()))
		encoded, _ := json.Marshal(result)
		return string(encoded)
	})
	js.Global().Set("openManusVerifyTrace", verify)
	select {}
}
