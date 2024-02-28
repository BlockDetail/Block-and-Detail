import { initCanvas } from "./canvasManager.js";
import { setupUI } from "./uiController.js";

// Initialization code
window.addEventListener("DOMContentLoaded", (event) => {
  initCanvas();
  setupUI();
});
