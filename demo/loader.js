var wasm_file = "./wasm/juggernaut_iris-c1d4446fddb02c61.wasm"; // wasm file
var wjs_file = "./wasm/juggernaut_iris-c1d4446fddb02c61.js"; // w.js file that links the wasm file

Module = {}
console.log("Loading webassembly version");

/// fetch wasm file and inject the js file
fetch(wasm_file)
  .then(response => response.arrayBuffer())
  .then(bytes => {
    Module.wasmBinary = bytes;
    console.log("wasm has loaded..");
    console.log("attaching as script");
    self.importScripts(wjs_file);
  });
