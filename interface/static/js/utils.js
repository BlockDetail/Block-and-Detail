export function loadImageOnCanvas(source, canvasId) {
  const canvas = document.getElementById(canvasId);
  const context = canvas.getContext("2d");
  const image = new Image();

  image.onload = function () {
    // Fit the image onto the canvas (you may want to adjust this to maintain aspect ratio)
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
  };

  image.src = source;
}

export function getFidelityValue() {
  var radios = document.getElementsByName('fid');
  var selectedFidelity;
  for (var radio of radios) {
    if (radio.checked) {
      selectedFidelity = radio.value;
      break;
    }
  }
  return selectedFidelity;
}

const strokeColors = {
  low: 'orange',
  mid: 'green',
  high: 'blue'
}
export function fidToStrokeColor(fidStr) {
  return strokeColors[fidStr];
}