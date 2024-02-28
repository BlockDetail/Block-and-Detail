/***
 * This file contains all the UI event listeners and handlers
 */

import {
  handleTextGenerate,
  getKeyImages,
  getGenImages,
  reCluster,
  handleSaveLayer,
  clearAllLayers,
  transformImageToStrokes,
  saveOverlay
} from "./apiService.js";
import {
  setupReferenceCanvas,
  toggleSelectionMode,
  toggleReference,
  undoStroke,
  redoStroke,
  toggleTranslateMode,
  toggleDrawCanvas,
  deleteSelectedStrokes,
  markConstruction,
} from "./canvasManager.js";
export function setupUI() {
  hideLoader();
  setupButtons();
  setupSliders();
  initializeUI();
}

export async function setCluster(cluster) {
  await fetch("/set-cluster", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ cluster: cluster }), //JSON.stringify({ layers: layers }),
  })
    .then((response) => response.json())
    .catch((error) => {
      console.error("Error:", error);
    });
}

export function updateGenImageList(listname, imageUrl, hide) {
  const list = document.getElementById(listname);
  const listItem = document.createElement("li");
  const image = new Image();
  image.height = 40;
  image.width = 40;
  image.src = imageUrl;
  if (hide === true) {
    console.log("hidden", imageUrl);
    image.style.filter = "brightness(0.3)";
  }
  listItem.appendChild(image);
  list.insertBefore(listItem, list.children[0]);
  image.addEventListener("mouseover", function () {
    fetch("/update-selected-ref-img", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ path: imageUrl }),
    }) // not waiting for response
    let im = document.getElementById('output_image');
    im.src = imageUrl;
    if (im.src.includes("reshape")){
      im.style.border = "solid 5px #FF0000";
    } else {
      im.style.border = "solid 5px #000000";
    }

    // add border to selected image
    // first remove border from all images from all lists
    let i = 0;
    let modes = ["reshape_image2", "image"];
    while (i < modes.length) {
      var listr = document.getElementById("genImageList" + i.toString());
      var children = listr.childNodes;
      children.forEach(function (item) {
        // console.log("removing", modes[i]+"clusterList", item)
        if (item.localName == "li") {
          item.firstChild.style.border = "none";
        }
      });
      i++;
    }
    // then add border to selected image
    if (image.src.includes("reshape")){
      image.style.border = "solid 2px #FF0000";
    } else {
      image.style.border = "solid 2px #000000";
    }

  });
  image.addEventListener("click", function () {
    fetch("/update-selected-ref-img", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ path: imageUrl }),
    }) // not waiting for response

    setupReferenceCanvas(imageUrl);
    let im = document.getElementById('output_image');
    im.src = imageUrl;
    if (im.src.includes("reshape")){
      im.style.border = "solid 5px #FF0000";
    } else {
      im.style.border = "solid 5px #000000";
    }

    // add border to selected image
    // first remove border from all images from all lists
    let i = 0;
    let modes = ["reshape_image2", "image"];
    while (i < modes.length) {
      var listr = document.getElementById("genImageList" + i.toString());
      var children = listr.childNodes;
      console.log(modes[i] + "clusterList", listr)
      children.forEach(function (item) {
        if (item.localName == "li") {
          item.firstChild.style.border = "none";
        }
      });
      i++;
    }
    // then add border to selected image
    if (image.src.includes("reshape")){
      image.style.border = "solid 2px #FF0000";
    } else {
      image.style.border = "solid 2px #000000";
    }

  });
}

export function updateKeyImageList(imageUrl) {
  fetch("/add-key", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ path: imageUrl }),
  })
    .then((response) => response.json())
    .catch((error) => {
      console.error("Error:", error);
    });
  const list = document.getElementById("keyImageList");
  const listItem = document.createElement("li");
  const image = new Image();
  image.src = imageUrl;
  listItem.appendChild(image);
  console.log("keyimagelist:", list);
  list.insertBefore(listItem, list.children[0]);
  image.addEventListener("click", function () {
    removeKeyImageList(imageUrl);
  });
}

export function updateImageList(imageUrl) {
  const list = document.getElementById("imageList");
  const listItem = document.createElement("li");
  const image = new Image();
  var x = document.createElement("INPUT");
  x.setAttribute("type", "text");
  x.setAttribute(
    "id",
    imageUrl.split("?")[0].split("/")[
    imageUrl.split("?")[0].split("/").length - 1
    ]
  );
  x.setAttribute("style", "width: 256px");
  console.log(
    "url input",
    imageUrl.split("?")[0].split("/")[
    imageUrl.split("?")[0].split("/").length - 1
    ]
  );
  x.className =
    "placeholder:italic placeholder:text-slate-400 block bg-white border border-slate-300 rounded-md py-2 pl-1 pr-1 shadow-sm focus:outline-none focus:border-sky-500 focus:ring-sky-500 focus:ring-1 sm:text-sm";
  image.src = imageUrl;
  listItem.appendChild(image);
  listItem.appendChild(x);
  list.insertBefore(listItem, list.children[0]);
  image.addEventListener("click", function () {
    updateKeyImageList(imageUrl);
  });
}

export async function removeKeyImageList(imageUrl) {
  await fetch("/remove-key", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ path: imageUrl }),
  })
    .then((response) => response.json())
    .catch((error) => {
      console.error("Error:", error);
    });
  const keylist = document.getElementById("keyImageList");
  console.log("target:", imageUrl);
  var children = keylist.childNodes;
  children.forEach(function (item) {
    if (item.localName == "li") {
      console.log("imageurl:", item.firstChild.src);
      if (item.firstChild.src.includes(imageUrl)) {
        console.log("removed imageurl:", item.firstChild.src);
        keylist.remove(item);
      }
    }
  });
}

export async function removeGenImageList(imageUrl) {
  await fetch("/remove-cluster", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ path: imageUrl }),
  })
    .then((response) => response.json())
    .catch((error) => {
      console.error("Error:", error);
    });
}

export function clearImageList() {
  const list = document.getElementById("imageList");
  if (list !== null) {
    while (list.firstChild) {
      list.removeChild(list.lastChild);
    }
    const keylist = document.getElementById("keyImageList");
    if (keylist !== null) {
      while (keylist.firstChild) {
        keylist.removeChild(keylist.lastChild);
      }
    }
  }
}

export function clearGenImageList() {
  let i = 0;
  let modes = ["reshape_image2", "image"];
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
    .getElementById("saveOverlayBtn")
    .addEventListener("click", saveOverlay);
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
    .getElementById("toggleRefBtn")
    .addEventListener("click", toggleReference);
  document
    .getElementById("overlayHEDBtn")
    .addEventListener("click", function () { toggleRefType("hed"); });
  document
    .getElementById("overlayImageBtn")
    .addEventListener("click", function () { toggleRefType("image"); });
  document
    .getElementById("overlayDilationBtn")
    .addEventListener("click", function () { setupReferenceCanvas("./static/sketch/dilation.png"); });
  document
    .getElementById("overlayContourDilationBtn")
    .addEventListener("click", function () { setupReferenceCanvas("./static/sketch/contour_dilation.png"); });
  document
    .getElementById("lcmBtn")
    .addEventListener("change", function () {
      if (document.getElementById("lcmBtn").checked) {
        document.getElementById("numsteps").value = 8;
        document.getElementById("controlnetscale").value = 1.0;
        document.getElementById("numsteps_val").innerText = 8;
        document.getElementById("controlnetscale_val").innerText = 1.0;
      } else {
        document.getElementById("numsteps").value = 40;
        document.getElementById("controlnetscale").value = 1.0;
        document.getElementById("numsteps_val").innerText = 40;
        document.getElementById("controlnetscale_val").innerText = 1.0;
      }
    });

  document.getElementById("toggleSelectionBtn").addEventListener("click", function () { toggleSelectionMode(true); });
  document.getElementById("exitSelectionBtn").addEventListener("click", function () { toggleSelectionMode(false); });
  document.getElementById("toggleTranslateBtn").addEventListener("click", toggleTranslateMode);


  document.getElementById("toggleSketchBtn").addEventListener("click", toggleDrawCanvas);
  document.getElementById("selectRefStrokeBtn").addEventListener("click", uiHandleSelectRefStroke);
}

function toggleRefType(type) {
  let im = document.getElementById('output_image');
  console.log("imsrc", im.src)
  if (im.src.includes("examples")) {
    if (type == "image") {
      console.log("wrong place");
      setupReferenceCanvas(im.src.replace("betterhed_", "gen_").replace("hed_", "gen_"));
    } else if (type == "hed") {
      console.log("imsrcreplace");
      console.log("imsrcreplace", im.src.replace("gen_", "hed_").replace("betterhed_", "hed_"));
      setupReferenceCanvas(im.src.replace("gen_", "hed_").replace("betterhed_", "hed_"));
    }
  }
}

async function uiHandleGenerate() {
  await handleSaveLayer().then(() => {
    // get fidelity
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
function uiHandleSelectRefStroke() {
  transformImageToStrokes();
}

function setupSliders() {
  const sliders = {
    numsteps: "numsteps_val",
    guidancescale: "guidancescale_val",
    controlnetscale: "controlnetscale_val",
    lod: "lod_val",
    num_images: "num_images_val",
    strength: "strength_val",
    dilation: "dilation_val",
    contour_dilation: "contour_dilation_val"
  };

  for (const [sliderId, displayId] of Object.entries(sliders)) {
    const slider = document.getElementById(sliderId);
    slider.addEventListener("change", () => {
      document.getElementById(displayId).innerText = slider.value;
      if (sliderId === "numcluster") reCluster();
      if (sliderId === "lod") getGenImages("hed");
    });
  }
}

function setupClusterButtons() {
  const clusterTypes = ["hed"];
  clusterTypes.forEach((type) => {
    document
      .getElementById(`${type.toUpperCase()}CBtn`)
      .addEventListener("click", () => setCluster(type));
  });
}

export function showGenImages(type) {
  clearGenImageList();
  getGenImages(type);
}

function setupKeydownListener() {
  document.addEventListener('keydown', function (event) {
    const key = event.key; // const {key} = event; ES6+
    if (key === "Backspace" || key === "Delete") {
      console.log("Deleting!")
      deleteSelectedStrokes();
    }
    if (!event.ctrlKey && key === "c") {
      markConstruction();
    }
    if (event.key === 't') {
      console.log("starting translation!")
      toggleTranslateMode();
    }
    if (event.ctrlKey && event.key === 'z') {
      console.log("Undo Stroke!")
      undoStroke();
    }
    if (event.ctrlKey && event.key === 'y') {
      console.log("Redo Stroke!")
      redoStroke();
    }
  });
}

function initializeUI() {
  clearAllLayers();
  getKeyImages();
  document.getElementById("drawingStatus").innerText = "Ready to Draw";
  document.getElementById("numStrokes").innerText =
    "Number of Strokes on Canvas: 0";
  setupKeydownListener();
}
