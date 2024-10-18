/***
 * This file contains the functions to manage the canvas
 */
let isDrawing = false; // decide if we record the stroke
let currentStroke = null; // single stroke
let currentLayer = []; // array of strokes

let isSelecting = false;  // Are we in selection mode?
let startTime, startX, startY; // Used to detect quick click on the canvas to exit selection mode
let selectionStart = null;  // Starting point of the selection rectangle
let selectionRect = null;  // Current dimensions of the selection rectangle
let selectedStrokes = [];  // Array to keep track of selected strokes
let isTranslating = false;  // Are we in translation mode?
let translationStart = null;  // Starting point for translation
let showDraw = true;
let showRef = false;
let currRef = null;

let refImg = "/static/dummy/empty.png"

let recentDeleted = [];
let isConstruction = true // in UI, we start with blocking mode
let currRefType = 0;

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

const referenceCanvas = document.getElementById("imageCanvas");
const referenceCtx = referenceCanvas.getContext("2d");

const hiddenCanvas = document.getElementById("hiddenCanvas");
const hiddenCtx = hiddenCanvas.getContext("2d");
const hiddenCanvas2 = document.getElementById("hiddenCanvas2");
const hiddenCtx2 = hiddenCanvas2.getContext("2d");
const hiddenCanvas3 = document.getElementById("hiddenCanvas3");
const hiddenCtx3 = hiddenCanvas3.getContext("2d");
const deleteButton = document.getElementById('deleteSelectedBtn');
const undoStrokeBtn = document.getElementById('undoStrokeBtn');
const redoStrokeBtn = document.getElementById('redoStrokeBtn');
const toggleSelectionBtn = document.getElementById('toggleSelectionBtn');

const canvasContainer = document.getElementById("outer-container");

/***
 * Public functions
 */

import {
  clearAllLayers,
} from "./apiService.js";

export function initCanvas() {
  setupDrawingCanvas();
  setupReferenceCanvas(refImg);
  document.getElementById("deleteSelectedBtn").addEventListener("click", deleteSelectedStrokes);
  document.getElementById("block").addEventListener('click', function () {
    console.log("Block mode");
    isConstruction = true;

    // if there is a selection, mark them as construction
    markConstruction();
  })
  document.getElementById("detail").addEventListener("click", function () {
    console.log("Detail mode");
    isConstruction = false;

    // if there is a selection, mark them as not construction
    notConstruction();
  });
}

export function returnRef() {
  return currRef;
}

export function refType() {
  return currRefType;
}

export function redoStroke() {
  if (recentDeleted.length != 0) {
    currentLayer.push(recentDeleted.pop());
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hiddenCtx.clearRect(0, 0, canvas.width, canvas.height);
    hiddenCtx2.clearRect(0, 0, canvas.width, canvas.height);
    hiddenCtx3.clearRect(0, 0, canvas.width, canvas.height);
    currentLayer.forEach(stroke => {
      drawStroke(stroke);
      drawHiddenStroke(stroke);
      if (stroke.color == "lime") {
        drawHiddenStroke2(stroke);
      }
      if (stroke.color == "black") {
        drawHiddenStroke3(stroke);
      }
    });
  }
}

export function undoStroke() {
  if (currentLayer.length > 0) {
    recentDeleted.push(currentLayer.pop());
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hiddenCtx.clearRect(0, 0, canvas.width, canvas.height);
    hiddenCtx2.clearRect(0, 0, canvas.width, canvas.height);
    hiddenCtx3.clearRect(0, 0, canvas.width, canvas.height);
    currentLayer.forEach(stroke => {
      drawStroke(stroke);
      drawHiddenStroke(stroke);
      if (stroke.color == "lime") {
        drawHiddenStroke2(stroke);
      }
      if (stroke.color == "black") {
        drawHiddenStroke3(stroke);
      }
    });
  }
}


export function clearStrokes() {
  currentLayer = [];
  currentStroke = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  clearAllLayers();
  redrawCanvas();
}

export function toggleReference() {
  showRef = !showRef;
  const btn = document.getElementById("toggleRefBtn");
  if (showRef) {
    setupReferenceCanvas(refImg);
    btn.innerHTML = "Hide Image";

  } else {
    clearRefLayer();
    btn.innerHTML = "Show Image";
  }
}

export function removeReference() {
  showRef = false
  const btn = document.getElementById("toggleRefBtn");
  clearRefLayer();
  btn.classList.add("hidden");
}

export function showReferenceCanvas() {
  showRef = true
}

function redrawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  currentLayer.forEach(drawStroke);

  hiddenCtx.clearRect(0, 0, hiddenCanvas.width, hiddenCanvas.height);
  currentLayer.forEach(drawHiddenStroke);

  hiddenCtx2.clearRect(0, 0, hiddenCanvas2.width, hiddenCanvas2.height);
  currentLayer.forEach(stroke => {
    if (stroke.color == "lime") {
      drawHiddenStroke2(stroke);
    }
  });

  hiddenCtx3.clearRect(0, 0, hiddenCanvas3.width, hiddenCanvas3.height);
  currentLayer.forEach(stroke => {
    if (stroke.color == "black") {
      drawHiddenStroke3(stroke);
    }
  });

  if (isSelecting || isTranslating) {
    highlightSelectedStrokes();
    // Draw the selection rectangle

    if (selectionRect) {
      ctx.strokeStyle = "rgba(0, 0, 255, 0.5)";
      ctx.strokeRect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
    }
  }

}

export function clearDrawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hiddenCtx.clearRect(0, 0, canvas.width, canvas.height);
  hiddenCtx2.clearRect(0, 0, canvas.width, canvas.height);
  hiddenCtx3.clearRect(0, 0, canvas.width, canvas.height);
}

export function toggleDrawCanvas(showDrawStatus) {

  showDraw = showDrawStatus;
  if (showDraw) {
    redrawCanvas();
  } else {
    clearDrawCanvas();
  }
}

export function setupReferenceCanvas(path) {
  if (!showRef) {
    toggleReference(); // we turn it on
  }
  showRef = true;

  console.log("Setting up reference canvas", path)

  const has_hidden = document.getElementById("toggleRefBtn").classList.contains("hidden");
  if (!path.includes("empty")) {
    // actual sample image
    if (has_hidden) {
      console.log("Enabling toggle ref button")
      document.getElementById("toggleRefBtn").classList.remove("hidden");
    }
  } else {
    if (!has_hidden) {
      document.getElementById("toggleRefBtn").classList.add("hidden");
      console.log("Disabling toggle ref button")
    }
  }


  refImg = path; // save the latest path for toggle

  // For reference
  referenceCtx.clearRect(0, 0, referenceCanvas.width, referenceCanvas.height);
  referenceCtx.globalAlpha = 0.3;
  const referenceImage = new Image();
  referenceImage.onload = function () {
    referenceCtx.drawImage(
      referenceImage,
      0,
      0,
      referenceCanvas.width,
      referenceCanvas.height
    );
  };
  referenceImage.src = path;
  currRef = path;
  let dirs = path.split("/");
  let filename = dirs[dirs.length - 1];
  console.log("refImage path", filename);
  let isFileRef = currRefType;
  if (filename.includes("gen")) {
    isFileRef = 1;

    // Add red border to the modal because this is naive result
    const modal_div = document.getElementById("modal");
    modal_div.classList.add("border-red-500");
    const modal_caption = document.getElementById("modal-caption");
    modal_caption.classList.add("text-red-500");

    // also add border to ref canvas
    canvasContainer.classList.add("outline-red-500");

  }
  if (filename.includes("reshape")) {
    isFileRef = 1;

    // Remove red border if it exists
    const modal_div = document.getElementById("modal");
    if (modal_div.classList.contains("border-red-500")) {
      modal_div.classList.remove("border-red-500");
    }
    const modal_caption = document.getElementById("modal-caption");
    if (modal_caption.classList.contains("text-red-500")) {
      modal_caption.classList.remove("text-red-500");
    }

    canvasContainer.classList.remove("outline-red-500");
  }
  currRefType = isFileRef;
}

export function setupCanvasSketch(path) {
  // For reference
  ctx.clearRect(0, 0, referenceCanvas.width, referenceCanvas.height);
  const referenceImage = new Image();
  referenceImage.onload = function () {
    ctx.drawImage(
      referenceImage,
      0,
      0,
      referenceCanvas.width,
      referenceCanvas.height
    );
  };
  referenceImage.src = path;
}

export function setupDrawingCanvasSketch(path) {
  // For sequence loading
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const referenceImage = new Image();
  referenceImage.onload = function () {
    ctx.drawImage(
      referenceImage,
      0,
      0,
      canvas.width,
      canvas.height
    );
  };
  referenceImage.src = path;
}


function changeText(button, staticVerb) {
  // Find the text node within the button
  const textNode = Array.from(button.childNodes).find(node => node.nodeType === Node.TEXT_NODE);
  console.log("textNode", textNode)
  if (textNode && textNode.textContent.trim() === staticVerb) {
    textNode.textContent = staticVerb + 'ing ';
  } else if (textNode && textNode.textContent.trim() === staticVerb + 'ing') {
    textNode.textContent = staticVerb;
  }
}

export function toggleSelectionMode() {
  isSelecting = !isSelecting;
  function turnOnSelectionMode() {
    selectedStrokes = []; // Clear the selected strokes when entering selection mode
    isDrawing = false; // Disable drawing mode when entering selection mode
    // Update the canvas cursor style
    canvas.style.cursor = isSelecting ? 'crosshair' : 'default';
  }
  function turnOffSelectionMode() {
    isDrawing = true; // Enable drawing mode when exiting selection mode
    selectedStrokes = []; // Clear the selected strokes when exiting selection mode
    canvas.style.cursor = isSelecting ? 'crosshair' : 'default';

    // also turn off translation mode
    isTranslating = false;
    canvas.style.cursor = 'default';

    // also turn off moving mode
    isTranslating = false;
    canvas.style.cursor = 'default';  // Default cursor when exiting translation mode
    selectionRect = null;

  }

  if (isSelecting) {
    turnOnSelectionMode();
  } else {
    turnOffSelectionMode();
  }

  redrawCanvas();

  changeText(document.getElementById("toggleSelectionBtn"), 'Select');
}


export function toggleTranslateMode() {
  if (isSelecting) {
    console.log("Toggle translate mode")
    isTranslating = !isTranslating;
    // Update button text based on the mode
    if (isTranslating) {
      canvas.style.cursor = 'grab';  // Hand cursor when in translation mode
    }
    else {
      canvas.style.cursor = 'default';  // Default cursor when exiting translation mode
    }
    if (!isTranslating) {
      selectionRect = null;
    }
    redrawCanvas();
  }
}

export function deleteSelectedStrokes() {
  if (selectedStrokes.length == 0) {
    alert("No strokes selected to delete! Please select strokes by clicking on select, and dragging a selection box around them.");
    return;
  }

  console.log("Deleting selected strokes: " + selectedStrokes.length)
  currentLayer = currentLayer.filter(stroke => !selectedStrokes.includes(stroke));
  recentDeleted = recentDeleted.concat(selectedStrokes);
  selectedStrokes = [];
  selectionRect = null;
  isTranslating = false;
  canvas.style.cursor = 'default';  // Default cursor when exiting translation mode
  redrawCanvas();
}

export function markConstruction() {
  // we press C in two sceneraios
  // 1. we are in selection mode, then we mark selected strokes as construction
  // 2. we are in drawing mode, then we mark current stroke as construction

  if (isSelecting) {
    selectedStrokes.forEach(stroke => {
      if (stroke.color != "lime") { stroke.color = "lime"; }
      else if (stroke.color == "lime") {
        stroke.color = "black";
      }
    });
    console.log("Marking selected as construction strokes: " + selectedStrokes.length)
    selectedStrokes = [];
    selectionRect = null;
    redrawCanvas();
  }

  if (currentStroke && currentStroke.color != "lime") {
    console.log("Marking current as construction stroke")
    currentStroke.color = "lime";
  }

  // we change the radio button to select "Block"
  const radioBtn = document.getElementById("block");
  radioBtn.checked = true;
  isConstruction = true;
}

export function markDetail() {
  if (isSelecting) {
    selectedStrokes.forEach(stroke => {
      if (stroke.color != "black") { stroke.color = "black"; }
      else if (stroke.color == "black") {
        stroke.color = "lime";
      }
    });
    console.log("Marking selected as detail strokes: " + selectedStrokes.length)
    selectedStrokes = [];
    selectionRect = null;
    redrawCanvas();
  }

  if (currentStroke && currentStroke.color != "black") {
    console.log("Marking current as detail stroke")
    currentStroke.color = "black";
  }


  const radioBtn = document.getElementById("detail");
  const blockBtn = document.getElementById("block");
  radioBtn.checked = true;
  blockBtn.checked = false;
  isConstruction = false;
}

export function notConstruction() {
  console.log("Marking selected as not construction strokes: " + selectedStrokes.length)
  currentLayer.forEach(stroke => {
    if (selectedStrokes.includes(stroke)) {
      stroke.color = "black";
    }
  });
  selectedStrokes = [];
  selectionRect = null;
  redrawCanvas();
}


/***
 * Private functions
 */


function setupDrawingCanvas() {
  canvas.addEventListener("mousedown", handleMouseDown);
  canvas.addEventListener("mousemove", handleMouseMove);
  canvas.addEventListener("mouseup", handleMouseUp);
  canvas.addEventListener("mouseleave", handleMouseLeave); // Optional, if you want to cancel the selection when the mouse leaves the canvas
}



function handleMouseDown(e) {
  startTime = Date.now();
  startX = e.clientX;
  startY = e.clientY;

  if (isTranslating && isStrokeInSelection({ points: [{ x: e.offsetX, y: e.offsetY }] })) {
    // Start translation if the click is within the selection rectangle
    translationStart = { x: e.offsetX, y: e.offsetY };
  } else if (isSelecting) {
    startSelection(e);
  } else {
    document.addEventListener('keydown', function (event) {
      const key = event.key; // const {key} = event; ES6+
      if (key === "c") {
        isConstruction = true;
      }
    })
    startDrawing(e);
  }
}

function handleMouseMove(e) {
  if (isTranslating && translationStart) {
    const deltaX = e.offsetX - translationStart.x;
    const deltaY = e.offsetY - translationStart.y;
    translateSelectedStrokes(deltaX, deltaY);
    // also translate the selection box
    selectionRect.x += deltaX;
    selectionRect.y += deltaY;

    translationStart = { x: e.offsetX, y: e.offsetY }; // Update for continuous translation
  } else if (isSelecting && selectionStart) {
    updateSelection(e);
  } else if (isDrawing && currentStroke) {
    continueDrawing(e);
  }
}

function handleMouseUp(e) {
  const endTime = Date.now();
  const endX = e.clientX;
  const endY = e.clientY;
  const timeDiff = endTime - startTime;
  const distance = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));

  // Thresholds to consider it a quick click (not a drag)
  const maxTime = 200; // milliseconds
  const maxDistance = 5; // pixels

  // Check if we clicked the delete button
  // Get the bounding box of the delete button
  const buttonsToCheck = [deleteButton, undoStrokeBtn, redoStrokeBtn, toggleSelectionBtn];
  for (const button of buttonsToCheck) {
    const buttonRect = button.getBoundingClientRect();
    const isClickOnButton =
      endX >= buttonRect.left &&
      endX <= buttonRect.right &&
      endY >= buttonRect.top &&
      endY <= buttonRect.bottom;
    if (isClickOnButton) {
      // Click was on the button, do nothing or return early
      console.log('Click was on the delete button.');
      return;
    }
  }

  if (timeDiff < maxTime && distance < maxDistance && isSelecting) {
    // It's a quick click, exit selection mode
    console.log('Quick click detected, removing selection box.');
    finishSelection();
  } else {
    translationStart = null;
    if (isSelecting) {
      finishSelection();
    } else if (isDrawing) {
      finishDrawing();
    }
  }
}
function handleMouseLeave() {
  finishDrawing();
}

function startDrawing(e) {
  console.log("Start drawing stroke", isConstruction)
  isDrawing = true;
  let color = "black";
  if (isConstruction) {
    color = "lime";
  }
  currentStroke = {
    points: [{ x: e.offsetX, y: e.offsetY }],
    color: color,
    width: ctx.lineWidth,
  };
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
}

function continueDrawing(e) {
  if (isDrawing && currentStroke) {
    const x = e.offsetX;
    const y = e.offsetY;
    currentStroke.points.push({ x, y });
    ctx.lineTo(x, y);
    ctx.strokeStyle = currentStroke.color;
    ctx.stroke();
  }
}

function finishDrawing() {
  if (isDrawing && currentStroke) {
    ctx.closePath();
    currentLayer.push(currentStroke);
    currentStroke = null;
    isDrawing = false;
    console.log("Finished drawing stroke: " + currentLayer.length)
    redrawCanvas();
  }
}

function drawStroke(stroke) {
  ctx.strokeStyle = stroke.color;
  ctx.lineWidth = stroke.width;
  ctx.beginPath();
  ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
  stroke.points.forEach(point => {
    ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();
}

function drawHiddenStroke(stroke) {
  hiddenCtx.strokeStyle = "black"
  hiddenCtx.lineWidth = stroke.width;
  hiddenCtx.beginPath();
  hiddenCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
  stroke.points.forEach(point => {
    hiddenCtx.lineTo(point.x, point.y);
  });
  hiddenCtx.stroke();
}

function drawHiddenStroke2(stroke) {
  hiddenCtx2.strokeStyle = "black"

  hiddenCtx2.lineWidth = stroke.width;
  hiddenCtx2.beginPath();
  hiddenCtx2.moveTo(stroke.points[0].x, stroke.points[0].y);
  stroke.points.forEach(point => {
    hiddenCtx2.lineTo(point.x, point.y);
  });
  hiddenCtx2.stroke();
}

function drawHiddenStroke3(stroke) {
  hiddenCtx3.strokeStyle = "black"

  hiddenCtx3.lineWidth = stroke.width;
  hiddenCtx3.beginPath();
  hiddenCtx3.moveTo(stroke.points[0].x, stroke.points[0].y);
  stroke.points.forEach(point => {
    hiddenCtx3.lineTo(point.x, point.y);
  });
  hiddenCtx3.stroke();
}


function clearRefLayer() {
  referenceCtx.clearRect(0, 0, referenceCanvas.width, referenceCanvas.height);
}


/***
 * SELECTION 
 */

function startSelection(e) {
  selectionStart = { x: e.offsetX, y: e.offsetY };
  selectionRect = { x: e.offsetX, y: e.offsetY, width: 0, height: 0 };
}

function updateSelection(e) {
  if (isTranslating) {
    return; // Don't update the selection rectangle if we are translating
  }
  if (selectionRect) {
    selectionRect.width = e.offsetX - selectionRect.x;
    selectionRect.height = e.offsetY - selectionRect.y;
    redrawCanvas(); // Redraw the canvas to show the selection rectangle
  }
}

function finishSelection() {
  // allow translate mode
  if (selectedStrokes.length > 0) {
    isTranslating = true;
    canvas.style.cursor = 'all-scroll';  // Hand cursor when in translation mode
    console.log("Turning on translation mode")
    // Show the delete button
    deleteButton.classList.remove('hidden');
  } else {
    // no selected strokes
    deleteButton.classList.add('hidden');
    canvas.style.cursor = 'crosshair';  // Default cursor when exiting translation mode
  }

  selectStrokes();
  selectionStart = null;
  redrawCanvas(); // Redraw the canvas to remove the selection rectangle
}


function selectStrokes() {
  selectedStrokes = currentLayer.filter(stroke => isStrokeInSelection(stroke));
  console.log("Selected strokes:", selectedStrokes.length)
  highlightSelectedStrokes();

  if (selectedStrokes.length == 0) {
    isTranslating = false;
    canvas.style.cursor = 'crosshair'; // crosshair cursor when in selection mode

    deleteButton.classList.add('hidden');
  } else {
    isTranslating = true;
    canvas.style.cursor = 'all-scroll';  // Hand cursor when in translation mode

    // add delete button
    deleteButton.classList.remove('hidden');
  }
}

function highlightSelectedStrokes() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  currentLayer.forEach(drawStroke);
  ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";  // Red color for highlighting
  ctx.lineWidth = 3;  // Slightly thicker line for highlight
  selectedStrokes.forEach(stroke => {
    ctx.beginPath();
    ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
    stroke.points.forEach(point => {
      ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
    ctx.closePath();
  });
  // change back to default
  ctx.strokeStyle = "rgba(0, 0, 0, 1)";
  ctx.lineWidth = 1;

}

function isPointInSelection(point) {
  // Check if the point is inside the selection rectangle
  const x = point.x;
  const y = point.y;
  if (!selectionRect) {
    return false;
  }
  const left = selectionRect.x;
  const right = selectionRect.x + selectionRect.width;
  const top = selectionRect.y;
  const bottom = selectionRect.y + selectionRect.height;

  return (left <= x && x <= right && top <= y && y <= bottom);
}

function getSideOfLine(point, lineStart, lineEnd) {
  // Check which side of the line the point is on
  // This is done by checking the sign of the cross product between the vectors (lineStart -> lineEnd) and (lineStart -> point)
  const x1 = lineStart.x;
  const y1 = lineStart.y;
  const x2 = lineEnd.x;
  const y2 = lineEnd.y;
  const x3 = point.x;
  const y3 = point.y;

  const crossProduct = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);

  return Math.sign(crossProduct);
}

function isLineIntersecting(line1, line2) {
  // Check if the two lines intersect
  // This is done by checking if the start and end points of the lines are on opposite sides of each other
  const { point1: line1Start, point2: line1End } = line1;
  const { point1: line2Start, point2: line2End } = line2;

  const line1StartSide = getSideOfLine(line1Start, line2Start, line2End);
  const line1EndSide = getSideOfLine(line1End, line2Start, line2End);

  const line2StartSide = getSideOfLine(line2Start, line1Start, line1End);
  const line2EndSide = getSideOfLine(line2End, line1Start, line1End);

  return (line1StartSide !== line1EndSide && line2StartSide !== line2EndSide);
}

function isLineCuttingSelection(point1, point2) {
  // Check if the line between point1 and point2 cuts through the selection rectangle
  // This is done by checking if the line intersects with any of the 4 sides of the rectangle
  const topLeft = { x: selectionRect.x, y: selectionRect.y };
  const topRight = { x: selectionRect.x + selectionRect.width, y: selectionRect.y };
  const bottomLeft = { x: selectionRect.x, y: selectionRect.y + selectionRect.height };
  const bottomRight = { x: selectionRect.x + selectionRect.width, y: selectionRect.y + selectionRect.height };

  const line = { point1, point2 };

  return (
    isLineIntersecting(line, { point1: topLeft, point2: topRight }) ||
    isLineIntersecting(line, { point1: topRight, point2: bottomRight }) ||
    isLineIntersecting(line, { point1: bottomRight, point2: bottomLeft }) ||
    isLineIntersecting(line, { point1: bottomLeft, point2: topLeft })
  );
}

function isStrokeInSelection(stroke) {
  if (!stroke || stroke.points.length === 0) {
    return false;
  }

  // Check if any point of the stroke is inside the selection rectangle
  const hasPointIn = stroke.points.some(point => {
    return isPointInSelection(point);
  })

  // check if the stroke cuts through the selection rectangle
  const hasCut = stroke.points.some((point, index) => {
    if (index === 0) {
      return false;
    }
    const prevPoint = stroke.points[index - 1];
    return isLineCuttingSelection(prevPoint, point);
  })

  return hasPointIn || hasCut;
}

function translateSelectedStrokes(deltaX, deltaY) {
  selectedStrokes.forEach(stroke => {
    stroke.points = stroke.points.map(point => ({
      x: point.x + deltaX,
      y: point.y + deltaY
    }));
  });
  redrawCanvas();
}
