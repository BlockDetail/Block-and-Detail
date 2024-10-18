/***
 * This file contains all the UI event listeners and handlers
 */

import {
  handleTextGenerate,
  getGenImages,
  handleSaveLayer,
  clearAllLayers,
} from "./apiService.js";
import {
  setupReferenceCanvas,
  toggleSelectionMode,
  undoStroke,
  redoStroke,
  deleteSelectedStrokes,
  markConstruction,
  clearStrokes,
  markDetail,
  showReferenceCanvas,
  toggleReference
} from "./canvasManager.js";
export function setupUI() {
  hideLoader();
  setupButtons();
  initializeUI();
}

var currentMode = "reshape" // for results display
var reshapeImageList = []
var originalImageList = []

export function updateImageResultList(resultType, images) {
  if (resultType == "reshape_image2") {
    reshapeImageList = images;
    console.log("reshapeImageList:", reshapeImageList);
  } else if (resultType == "image") {
    originalImageList = images;
    console.log("originalImageList:", originalImageList);
  }
}

export function updateGenImageList(imageUrl, curSampleIdx) {
  const imageHtmlID = "res_" + curSampleIdx.toString() + "_img";
  // Append a unique query parameter (e.g., current timestamp) to force reload
  const updatedUrl = `${imageUrl}?t=${new Date().getTime()}`;
  const old_image = document.getElementById(imageHtmlID);
  const new_image = old_image.cloneNode(true);
  old_image.parentNode.replaceChild(new_image, old_image);
  new_image.src = updatedUrl;
  if (updatedUrl.includes("gen")) { // gen
    new_image.classList.add("border-red-500")
  }
  if (updatedUrl.includes("reshape2")) { // reshape2
    if (new_image
      .classList.contains("border-red-500")) {
      new_image.classList.remove("border-red-500")
    }
  }
  new_image.addEventListener('click', function () {
    const imgSrc = this.src;
    fetch("/update-selected-ref-img-selected", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ path: imgSrc }),
    }
    )
    showReferenceCanvas()
    setupReferenceCanvas(imgSrc);
    console.log("Updated reference image:", imgSrc);
  });

}

export function clearGenImageList() {
  let i = 0;
  let modes = ["reshape_image2"]//, "image"];
  while (i < modes.length) {
    const list0 = document.getElementById(modes[i] + "clusterList");
    while (list0.firstChild) {
      list0.removeChild(list0.lastChild);
    }
    i++;
  }
}

/***
 * Private functions
 */

function hideLoader() {
  document.getElementsByClassName("loader")[0].style.display = "none";
}


function setupButtons() {
  document
    .getElementById("undoStrokeBtn")
    .addEventListener("click", undoStroke);
  document
    .getElementById("redoStrokeBtn")
    .addEventListener("click", redoStroke);
  document
    .getElementById("generateBtn")
    .addEventListener("click", uiHandleGenerate);
  document
    .getElementById("toggleRefBtn").addEventListener('click', toggleReference);

  document.getElementById("toggleSelectionBtn").addEventListener("click", toggleSelectionMode);
  document.getElementById("advanced-setting-btn").addEventListener("click", function () {
    document.getElementById("advanced-setting-wrapper").classList.toggle("hidden");
  });

  document.getElementById("new-sketch-btn").addEventListener("click", function () {
    clearStrokes();
  })
}

function toggleImageModes(mode) {
  var imageList = []
  currentMode = mode
  console.log("currentMode:", currentMode)
  const modalCaption = document.getElementById('modal-caption');
  if (mode == "original") {
    imageList = originalImageList
    modalCaption.innerHTML = "First pass tight control"
    modalCaption.classList.add("text-red-500")
  }
  else if (mode == "reshape") {
    imageList = reshapeImageList
    modalCaption.innerHTML = "Block and detail"
    if (modalCaption.classList.contains("text-red-500")) {
      modalCaption.classList.remove("text-red-500")
    }
  }
  console.log("imageList:", imageList)
  const listLen = imageList.length
  for (var i = 0; i < listLen; i++) {
    updateGenImageList(imageList[i], i)
  }

  const modal_image = document.getElementById("modal-image");
  const path = modal_image.src;
  // modal image src is always ..._{sample_idx}.png, the last underscore is the split point
  const match = path.match(/(gen|reshape2)_(\d+)\.png/);  // Match until `.png`, ignore anything after
  if (match == null) {
    return;
  }
  if (match && match[2]) {
    const number = parseInt(match[2], 10);
    console.log("model_idx:", number)
    const new_src = imageList[number];
    modal_image.src = new_src;
    console.log("Updated modal image src:", new_src)
    setupReferenceCanvas(new_src);
  }
}

async function uiHandleGenerate() {
  await handleSaveLayer().then(() => {
    const modal = document.getElementById('modal');
    modal.classList.remove('show');
    var radios = document.getElementsByName('model');
    var selectedModel;
    for (var radio of radios) {
      if (radio.checked) {
        selectedModel = radio.value;
        break;
      }
    }
    handleTextGenerate(selectedModel);
  });
}

export function showGenImages() {
  getGenImages();
}

function setupKeydownListener() {
  document.addEventListener('keydown', function (event) {
    const activeElement = document.activeElement;
    const isTextInput = activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA';

    // If typing in an input or textarea, don't handle the keydown
    if (isTextInput) {
      return;
    }

    const key = event.key; // const {key} = event; ES6+
    if (key === "Backspace" || key === "Delete") {
      console.log("Deleting!")
      deleteSelectedStrokes();
    }
    if (key == "u") {
      undoStroke();
    }
    if (key == "r") {
      redoStroke();
    }
    if (!event.ctrlKey && key === "b") {
      markConstruction();
    }
    if (!event.ctrlKey && key === "d") {
      markDetail();
    }
    // cmd + 1 is for toggle image mode
    if (key === "1") {
      toggleImageModes("original");
    };
    if (key === "2") {
      toggleImageModes("reshape");
    }
    if (key == "s") {
      toggleSelectionMode();
    }
  })
}

function initializeUI() {
  clearAllLayers();
  setupKeydownListener();
}
