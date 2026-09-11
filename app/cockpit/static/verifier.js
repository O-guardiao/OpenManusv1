(function () {
  "use strict";

  let wasmReady = null;

  function loadVerifier() {
    if (wasmReady) return wasmReady;
    wasmReady = (async function () {
      if (typeof window.Go !== "function") {
        throw new Error("runtime Go/WASM não foi construído");
      }
      const go = new window.Go();
      const response = await fetch("/assets/verifier.wasm", { cache: "no-store" });
      if (!response.ok) throw new Error("verifier.wasm indisponível");
      const bytes = await response.arrayBuffer();
      const instance = await WebAssembly.instantiate(bytes, go.importObject);
      void go.run(instance.instance);
      for (let attempts = 0; attempts < 100; attempts += 1) {
        if (typeof window.openManusVerifyTrace === "function") return;
        await new Promise((resolve) => window.setTimeout(resolve, 10));
      }
      throw new Error("verificador WASM não inicializou");
    })();
    return wasmReady;
  }

  document.addEventListener("click", async function (event) {
    const button = event.target.closest("[data-verify-trace]");
    if (!button) return;
    const output = button.closest(".task-card").querySelector(".wasm-result");
    output.textContent = "Verificando…";
    try {
      await loadVerifier();
      const response = await fetch(button.dataset.exportUrl, { cache: "no-store" });
      if (!response.ok) throw new Error("não foi possível obter a prova");
      const result = JSON.parse(window.openManusVerifyTrace(await response.text()));
      output.textContent = result.valid
        ? `Válida: ${result.event_count} eventos, head ${result.head_hash.slice(0, 12)}`
        : `Inválida: ${result.errors.join("; ")}`;
      output.dataset.valid = String(result.valid);
    } catch (error) {
      output.textContent = `WASM indisponível: ${error.message}`;
      output.dataset.valid = "false";
    }
  });
})();
