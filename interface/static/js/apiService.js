/***
 * This file contains the functions that make requests to the Flask API
 */

import {
  updateGenImageList,
  showGenImages,
  updateImageResultList
} from "./uiController.js";
import { setupReferenceCanvas, returnRef, refType } from "./canvasManager.js";
const canvas = document.getElementById("hiddenCanvas");
const constructionCanvas = document.getElementById("hiddenCanvas2");
const detailCanvas = document.getElementById("hiddenCanvas3");

var success = new Audio('static/dummy/success.wav');
var failure = new Audio('static/dummy/failure.wav');

let genRound = 0;
const NUM_GEN_SAMPLES = 4;

export async function handleTextGenerate(selectedModel) {
  var promptText = "";
  promptText = document.getElementById("promptInput").value;
  console.log("in handle image generate", promptText, " trimmed:", promptText.trim());

  await clearAllLayers();
  let drawid = await handleSaveLayer();
  let blocking = false;
  let scheduler = "euler";
  let color_consistency = 0;
  if (document.getElementById("colorconBtn").checked) {
    color_consistency = 1;
  }
  let selectedRef = refType();
  let refImage = returnRef();
  if (color_consistency == 1 && selectedRef == 0 && genRound != 0) {
    document.getElementById("warning").innerHTML = "!Ref Image Not Selected for Color Consistency!";
  } else {
    if (promptText.trim()) {
      document.getElementById("warning").innerHTML = "";
      document.getElementById("generateBtn").style.color = "red";
      document.getElementById("generateBtn").innerHTML = "......";
      document.getElementById('loader-overlay').classList.remove('hidden');
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
          num_samples: NUM_GEN_SAMPLES,  //document.getElementById("num_images").value,
          guidancescale: document.getElementById("guidancescale").value,
          numsteps: document.getElementById("numsteps").value,
          controlnetscale: document.getElementById("controlnetscale").value,
          blocking: blocking,
          strength: document.getElementById("strength_val").value,
          dilation: document.getElementById("dilationOutput").value,
          contour_dilation: document.getElementById("detailOutput").value,
          seed: document.getElementById("seed").value,
          color_consistency: color_consistency,
          ref_image: refImage,
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          // Handle the response data which should contain the generated images
          if (data["generating"] == 0) {
            showGenImages();
            setupReferenceCanvas("./static/dummy/empty.png?" + Date.now());
          }
          console.log(data);
        })
        .then((data) => {
          document.getElementById("generateBtn").style.color = "white";
          document.getElementById("generateBtn").innerHTML = "Generate!";
          success.play();
          genRound = genRound + 1;
          document.getElementById('loader-overlay').classList.add('hidden');
        })
        .catch((error) => {
          failure.play();
          console.error("Error:", error);
        });
    } else {
      document.getElementById("warning").innerHTML = "!No prompt entered!";
    }
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
    body: JSON.stringify({ data: pixelData, detail: detailData, construction: constructionData }),
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

export async function getGenImages() {
  let selectedVisMode = "";
  // clearGenImageList();
  let i = 0;
  let modes = ["reshape_image2", "image"];
  while (i < modes.length) {
    selectedVisMode = modes[i];
    await fetch("/get-gen-images", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        type: "betterhed", // not sure what this is for
        num_cluster: 1,
        fid: "low", // placeholder to not break the API
        vis_mode: selectedVisMode,
      }),
    })
      .then((response) => {
        return response.json();
      })
      .then((images) => {
        updateImageResultList(selectedVisMode, images["cluster0"]);

        if (selectedVisMode == "reshape_image2") {
          // our results
          images["labels"].forEach((clusterid) => {
            let numSamples = images["cluster0"][0].length;
            for (let j = 0; j < numSamples; j++) {
              updateGenImageList(
                images["cluster0"][j] + "?" + Date.now(),
                j
              );
            }
          });
        }
      })
      .catch((error) => {
        console.error("Error:", error);
      });
    i++;
  }
}

export async function clearAllLayers() {
  await fetch("/clear-all")
    .then((response) => response.json())
    .catch((error) => {
      console.error("Error:", error);
    });
}

