package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/O-guardiao/OpenManusv1/cockpit/verifier"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: oaktrace-verify <trace.json>")
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	result := verifier.VerifyJSON(data)
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if !result.Valid {
		os.Exit(1)
	}
}
