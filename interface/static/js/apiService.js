/***
 * This file contains the functions that make requests to the Flask API
 */

import {
  updateImageList,
  updateKeyImageList,
  clearGenImageList,
  updateGenImageList,
  showGenImages
} from "./uiController.js";
import { setupReferenceCanvas, clearStrokes, setupCanvasSketch, setChanged, setupDrawingCanvasSketch, getCompletionBoxData, getCompletionBoxesData, returnRef } from "./canvasManager.js";
const canvas = document.getElementById("hiddenCanvas");
const constructionCanvas = document.getElementById("hiddenCanvas2");
const detailCanvas = document.getElementById("hiddenCanvas3");

var success = new Audio('static/dummy/success.wav');
var failure = new Audio('static/dummy/failure.wav');

export async function handleTextGenerate(selectedModel) {
  var promptText = "";
  let boxes = getCompletionBoxesData();
  promptText = document.getElementById("promptInput").value;
  console.log("in handle image generate", promptText, promptText.trim());

  await clearAllLayers();
  let drawid = await handleSaveLayer();
  let blocking = false;
  let scheduler = "euler";
  if (document.getElementById("lcmBtn").checked) {
    scheduler = "lcm";
  }
  if (promptText.trim()) {
    document.getElementById("warning").innerHTML = "";
    console.log("generate button", document.getElementById("generateBtn"));
    document.getElementById("generateBtn").style.color = "red";
    document.getElementById("generateBtn").innerHTML = "......";
    setChanged(false);
    await fetch("/generate-images", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: promptText,
        model: selectedModel,
        scheduler: scheduler,
        num_cluster: 1,
        num_samples: document.getElementById("num_images").value,
        guidancescale: document.getElementById("guidancescale").value,
        numsteps: document.getElementById("numsteps").value,
        controlnetscale: document.getElementById("controlnetscale").value,
        blocking: blocking,
        strength: document.getElementById("strength").value,
        dilation: document.getElementById("dilation").value,
        contour_dilation: document.getElementById("contour_dilation").value,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Handle the response data which should contain the generated images
        clearGenImageList();
        if (data["generating"] == 0) {
          showGenImages("betterhed");
          setupReferenceCanvas("./static/dummy/empty.png?" + Date.now());
        }
        console.log(data);
      })
      .then((data) => {
        document.getElementById("generateBtn").style.color = "black";
        document.getElementById("generateBtn").innerHTML = "GO!";
        success.play();
        var elm = document.getElementById('autosaveBtn');
        if (elm.checked) {
          saveOverlay();
        }
      })
      .catch((error) => {
        failure.play();
        console.error("Error:", error);
      });
  } else {
    document.getElementById("warning").innerHTML = "!No prompt entered!";
  }
}

export async function handleSaveLayer() {
  // Convert canvas to pixel data
  let drawid = "";
  const pixelData = canvas.toDataURL("image/jpg");
  const constructionData = constructionCanvas.toDataURL("image/jpg");
  const detailData = detailCanvas.toDataURL("image/jpg");
  await fetch("/save-drawing", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ data: pixelData, detail: detailData, construction: constructionData }), //JSON.stringify({ layers: layers }),
  })
    .then((response) => response.json())
    .then((data) => {
      drawid = data["drawid"];
      console.log("drawid in handleSaveLayer", drawid);
      console.log("Drawing saved:", data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
  return drawid;
}

export function getImages() {
  fetch("/get-images")
    .then((response) => response.json())
    .then((images) => {
      images.forEach((imageUrl) => {
        updateImageList(imageUrl + "?" + Date.now());
      });
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

export async function getGenImages(type) {
  let selectedVisMode = "";
  clearGenImageList();
  let i = 0;
  let modes = ["reshape_image2", "image"];
  while (i < modes.length) {
    selectedVisMode = modes[i];
    if (selectedVisMode.includes("gaussian")) {
      type = "image";
    }
    console.log("type", type);
    await fetch("/get-gen-images", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        type: type,
        num_cluster: 1,
        fid: "low", // placeholder to not break the API
        vis_mode: selectedVisMode,
      }),
    })
      .then((response) => {
        return response.json();
      })
      .then((images) => {
        console.log("hidden list", images["hide"]);
        console.log("current list name: ", selectedVisMode + "clusterList");
        const list = document.getElementById(selectedVisMode + "clusterList");
        console.log("apiservice", selectedVisMode + "clusterList", list)
        images["labels"].forEach((clusterid) => {
          const divItem = document.createElement("div");
          const ulItem = document.createElement("ul");
          divItem.style.width = "180px";
          divItem.style.display = "inline-block";
          divItem.style.verticalAlign = "top";
          ulItem.id = "genImageList" + i;
          divItem.appendChild(ulItem);
          list.appendChild(divItem);
          console.log("current list images: ", i, images["cluster0"][0]);
          images["cluster0"][0].forEach((imageUrl) => {
            console.log("curr image", imageUrl);
            updateGenImageList(
              "genImageList" + i,
              imageUrl + "?" + Date.now(),
              images["hide"].includes(imageUrl)
            );
          });
        });
        console.log("images: ", images);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
    i++;
  }
}

export async function reCluster() {
  clearGenImageList();
  await fetch("/re-cluster", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      num_cluster: 1,
    }),
  })
    .then((response) => response.json())
    .then((images) => {
      getGenImages("image");
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

export async function clearAllLayers() {
  await fetch("/clear-all")
    .then((response) => response.json())
    .catch((error) => {
      console.error("Error:", error);
    });
}

export async function saveOverlay() {
  await fetch("/save-overlay", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url: returnRef(), prompt: document.getElementById("promptInput").value }),
  })
    .then((response) => {
      let data = response.json();
      return data;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

export async function transformImageToStrokes() {
  console.log("in transform image to strokes");
  await fetch("/transform-img-to-strokes", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ test: "hello" }),
  })
    .then((response) => {
      return response.json();
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
