import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

// No StrictMode: this app drives an imperative WebGL scatterplot (regl) and a
// dev __demo hook; StrictMode's double-mount would create/destroy two GL
// contexts and duplicate listeners.
createRoot(document.getElementById("root")).render(<App />);
