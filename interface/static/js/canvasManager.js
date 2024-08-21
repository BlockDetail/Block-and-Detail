/***
 * This file contains the functions to manage the canvas
 */
let isDrawing = false; // decide if we record the stroke
let isChanged = false;
let currentStroke = null; // single stroke
let currentLayer = []; // array of strokes

let isSelecting = false;  // Are we in selection mode?
let selectionStart = null;  // Starting point of the selection rectangle
let selectionRect = null;  // Current dimensions of the selection rectangle
let selectedStrokes = [];  // Array to keep track of selected strokes
let isTranslating = false;  // Are we in translation mode?
let translationStart = null;  // Starting point for translation
let showDraw = true;

let currRef = null;

let refImg = "/static/dummy/empty.png"
let showRef = false;

let recentDeleted = [];
let isConstruction = false
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

// selecting for completion
let isSelectingCompletionBox = false; // Are we creating a completion box?
let selectionCompletionBoxStart = null;  // Starting point of the selection rectangle
let selectionCompletionBoxRect = null;  // Current dimensions of the selection rectangle
let completionBoxes = [];  // Array to keep track of all completion boxes
let completionInputElems = []; // Array to keep track of all completion input elements

let colors = ["rgba(69, 255, 87, 0.7)",
  "rgba(161, 73, 197, 0.7)",
  "rgba(200, 0, 19, 0.7)",
  "rgba(39, 170, 255, 0.7)",
  "rgba(0, 234, 234, 0.7)",
  "rgba(236, 153, 0, 0.7)",
  "rgba(108, 190, 244, 0.7)",
  "rgba(0, 230, 151, 0.7)",
  "rgba(160, 232, 27, 0.7)",
  "rgba(127, 127, 127, 0.7)",
  "rgba(100, 50, 255, 0.7)",
  "rgba(255, 50, 145, 0.7)",]

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
  document.getElementById("constructionBtn").addEventListener("click", markConstruction);
  document.getElementById("notConstructionBtn").addEventListener("click", notConstruction);
}
export function layerLength() {
  return currentLayer.length;
}

export function isCurrDrawing() {
  return isDrawing;
}

export function checkChanged() {
  return isChanged;
}

export function setChanged(val) {
  isChanged = val;
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

export function clearLayer() {
  currentLayer = [];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}


export function clearStrokes() {
  currentLayer = [];
  currentStroke = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  clearAllLayers();
}


export function toggleReference() {
  showRef = !showRef;
  if (showRef) {
    setupReferenceCanvas(refImg);
    document.getElementById("toggleRefBtn").innerHTML = '👁';
  } else {
    clearRefLayer();
    document.getElementById("toggleRefBtn").innerHTML = '👁';
  }
}

function redrawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  currentLayer.forEach(drawStroke);
  // console.log("ctx strokes", currentLayer);

  hiddenCtx.clearRect(0, 0, hiddenCanvas.width, hiddenCanvas.height);
  currentLayer.forEach(drawHiddenStroke);
  // console.log("hiddenctx strokes", currentLayer);

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

  if (isSelectingCompletionBox) {
    if (selectionCompletionBoxRect) {
      ctx.strokeStyle = "rgba(0, 255, 0, 0.5)";
      ctx.strokeRect(selectionCompletionBoxRect.x, selectionCompletionBoxRect.y, selectionCompletionBoxRect.width, selectionCompletionBoxRect.height);

    }
  }

  let i = 0;

  completionBoxes.forEach(box => {
    if (i > colors.length) {
      i = 0;
    }
    ctx.strokeStyle = colors[i];
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    i++;
  })
}

export function clearDrawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hiddenCtx.clearRect(0, 0, canvas.width, canvas.height);
  hiddenCtx2.clearRect(0, 0, canvas.width, canvas.height);
  hiddenCtx3.clearRect(0, 0, canvas.width, canvas.height);
}

export function toggleDrawCanvas() {

  showDraw = !showDraw;
  if (showDraw) {
    redrawCanvas();
  } else {
    clearDrawCanvas();
  }
}

export function setupReferenceCanvas(path) {
  showRef = true;
  document.getElementById("toggleRefBtn").innerHTML = '👁';

  refImg = path; // save the latest path for toggle

  // For reference
  referenceCtx.clearRect(0, 0, referenceCanvas.width, referenceCanvas.height);
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
  referenceCtx.globalAlpha = 0.3;
  referenceImage.src = path;
  currRef = path;
  let dirs = path.split("/");
  let filename = dirs[dirs.length - 1];
  console.log("refImage path", filename);
  let isFileRef = currRefType;
  if (filename.includes("gen")){
    isFileRef = 1;
  }
  if (filename.includes("reshape")){
    isFileRef = 1;
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

export function toggleSelectionMode(select) {
  isSelecting = select;

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
    document.getElementById("toggleTranslateBtn").innerHTML = 'Translate Selected Strokes';
    canvas.style.cursor = 'default';
  }

  if (isSelecting) {
    turnOnSelectionMode();
  } else {
    turnOffSelectionMode();
  }

  redrawCanvas();
}

export function toggleTranslateMode() {
  isTranslating = !isTranslating;
  // Update button text based on the mode
  document.getElementById("toggleTranslateBtn").innerHTML = isTranslating ? 'Exit Translate Mode' : 'Translate Selected Strokes';
  // Update the canvas cursor style based on the mode
  if (isTranslating) {
    canvas.style.cursor = 'grab';  // Hand cursor when in translation mode
  } else {
    canvas.style.cursor = 'default';  // Default cursor when exiting translation mode
  }
  if (!isTranslating) {
    selectionRect = null;
  }
  redrawCanvas();
}

export function deleteSelectedStrokes() {
  console.log("Deleting selected strokes: " + selectedStrokes.length)
  currentLayer = currentLayer.filter(stroke => !selectedStrokes.includes(stroke));
  selectedStrokes = [];
  selectionRect = null;
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

export function imageToStroke(img) {

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
  if (isTranslating && isStrokeInSelection({ points: [{ x: e.offsetX, y: e.offsetY }] })) {
    // Start translation if the click is within the selection rectangle
    translationStart = { x: e.offsetX, y: e.offsetY };
  } else if (isSelecting) {
    startSelection(e);
  } else if (isSelectingCompletionBox) {
    startCompletionBoxSelection(e)
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
  } else if (isSelectingCompletionBox && selectionCompletionBoxStart) {
    updateCompletionBoxSelection(e);
  } else if (isDrawing && currentStroke) {
    continueDrawing(e);
  }
}

function handleMouseUp() {
  translationStart = null;
  if (isSelecting) {
    finishSelection();
  } else if (isSelectingCompletionBox) {
    finishCompletionBoxSelection();
  } else if (isDrawing) {
    finishDrawing();
    isChanged = true;
  }
}
function handleMouseLeave() {
  finishDrawing();
}

function startDrawing(e) {
  document.getElementById("drawingStatus").innerHTML = '<font size="1">Drawing!</font>';
  isDrawing = true;
  currentStroke = {
    points: [{ x: e.offsetX, y: e.offsetY }],
    color: "black",
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
    isConstruction = false;
    currentLayer.push(currentStroke);
    currentStroke = null;
    isDrawing = false;
    recentDeleted = [];
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

function drawRefStroke(stroke) {
  referenceCtx.strokeStyle = stroke.color;
  referenceCtx.lineWidth = stroke.width;
  referenceCtx.beginPath();
  referenceCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
  stroke.points.forEach(point => {
    referenceCtx.lineTo(point.x, point.y);
  });
  referenceCtx.stroke();
}

function clearRefLayer() {
  referenceCtx.clearRect(0, 0, referenceCanvas.width, referenceCanvas.height);
}

/***
 * COMPLETION BOX
 */

function startCompletionBoxSelection(e) {
  selectionCompletionBoxStart = { x: e.offsetX, y: e.offsetY };
  selectionCompletionBoxRect = { x: e.offsetX, y: e.offsetY, width: 0, height: 0 };
  console.log("Start Completion Box Selection")
}

function updateCompletionBoxSelection(e) {
  selectionCompletionBoxRect.width = e.offsetX - selectionCompletionBoxRect.x;
  selectionCompletionBoxRect.height = e.offsetY - selectionCompletionBoxRect.y;
  redrawCanvas(); // Redraw the canvas to show the selection rectangle
  addConfirmBoxToCanvas();
}

function finishCompletionBoxSelection() {
  selectionCompletionBoxStart = null;
  redrawCanvas(); // Redraw the canvas to remove the selection rectangle
}

export function getCompletionBoxData() {
  return selectionCompletionBoxRect;
}

export function getCompletionBoxesData() {
  return completionBoxes;
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
  selectionRect.width = e.offsetX - selectionRect.x;
  selectionRect.height = e.offsetY - selectionRect.y;
  redrawCanvas(); // Redraw the canvas to show the selection rectangle
}

function finishSelection() {
  selectStrokes();
  selectionStart = null;
  redrawCanvas(); // Redraw the canvas to remove the selection rectangle
}


function selectStrokes() {
  selectedStrokes = currentLayer.filter(stroke => isStrokeInSelection(stroke));
  console.log("Selected strokes:", selectedStrokes.length)
  highlightSelectedStrokes();
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

function handleConfirmBoxSubmit() {
  console.log("Fixing the box")
  completionBoxes.push(selectionCompletionBoxRect);
  let boxIdx = completionBoxes.length - 1;

  // add prompt box for completion
  addTextBoxToCanvasForCompletion(boxIdx);

  // cleanup
  finishCompletionBoxSelection();
  selectionCompletionBoxRect = null;
  redrawCanvas();

  // remove UI
  var container = document.getElementById('confirmBoxesContainer');
  container.innerHTML = "";

  console.log("Currently have " + completionBoxes.length + " boxes")
}

function addConfirmBoxToCanvas() {
  if (document.getElementById('confirmBoxesContainer')) {
    // remove it
    document.getElementById('confirmBoxesContainer').innerHTML = "";
  }

  // Create a button around mouse coordinates
  const container = document.getElementById('confirmBoxesContainer');
  var confirmButton = document.createElement('button');
  confirmButton.innerText = 'Confirm Box';
  confirmButton.onclick = function () {
    handleConfirmBoxSubmit();
  };
  var buttonContainer = document.createElement('div');
  buttonContainer.className = "absolute border rounded bg-lime-200"
  buttonContainer.style.left = selectionCompletionBoxRect.x + selectionCompletionBoxRect.width + 'px';
  buttonContainer.style.top = selectionCompletionBoxRect.y + selectionCompletionBoxRect.height + 'px';
  buttonContainer.style.zIndex = 100;  // Make sure the text box is on top of the canvas

  buttonContainer.appendChild(confirmButton);
  container.appendChild(buttonContainer);
}