const faceapi = require("face-api.js")  
const canvas = require("canvas")  
const fs = require("fs")  
const path = require("path")

// mokey pathing the faceapi canvas
const { Canvas, Image, ImageData } = canvas  
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })

const faceDetectionNet = faceapi.nets.ssdMobilenetv1

// SsdMobilenetv1Options
const minConfidence = 0.5

// TinyFaceDetectorOptions
const inputSize = 408  
const scoreThreshold = 0.5

// MtcnnOptions
const minFaceSize = 50  
const scaleFactor = 0.8

function getFaceDetectorOptions(net) {  
    return net === faceapi.nets.ssdMobilenetv1
        ? new faceapi.SsdMobilenetv1Options({ minConfidence })
        : (net === faceapi.nets.tinyFaceDetector
            ? new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold })
            : new faceapi.MtcnnOptions({ minFaceSize, scaleFactor })
        )
}

const faceDetectionOptions = getFaceDetectorOptions(faceDetectionNet)

// simple utils to save files
const baseDir = path.resolve(__dirname, './out')  
function saveFile(fileName, buf) {  
    if (!fs.existsSync(baseDir)) {
      fs.mkdirSync(baseDir)
    }
    // this is ok for prototyping but using sync methods
    // is bad practice in NodeJS
    fs.writeFileSync(path.resolve(baseDir, fileName), buf)
  }


//https://github.com/justadudewhohacks/face-api.js/#face-recognition-by-matching-descriptors
async function run() {

    // load weights
    await faceDetectionNet.loadFromDisk('weights')
    await faceapi.nets.faceLandmark68Net.loadFromDisk('weights')
    await faceapi.nets.faceRecognitionNet.loadFromDisk('weights')

    const referenceImage = await canvas.loadImage('imgs_ref/humberto.jpg')
    const queryImage = await canvas.loadImage('imgs_src/query.jpg')

    const resultsRef = await faceapi.detectAllFaces(referenceImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors()

    const faceMatcher = new faceapi.FaceMatcher(resultsRef)

  const resultsQuery = await faceapi.detectAllFaces(queryImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors()
    
    const labels = faceMatcher.labeledDescriptors
    .map(ld => ld.label)
  const refDrawBoxes = resultsRef
    .map(res => res.detection.box)
    .map((box, i) => new faceapi.draw.DrawBox(box, { label: labels[i] }))
  const outRef = faceapi.createCanvasFromMedia(referenceImage)
  refDrawBoxes.forEach(drawBox => drawBox.draw(outRef))

  saveFile('referenceImage.jpg', (outRef).toBuffer('image/jpeg'))

  const queryDrawBoxes = resultsQuery.map(res => {
    const bestMatch = faceMatcher.findBestMatch(res.descriptor)
    return new faceapi.draw.DrawBox(res.detection.box, { label: bestMatch.toString() })
  })
  const outQuery = faceapi.createCanvasFromMedia(queryImage)
  queryDrawBoxes.forEach(drawBox => drawBox.draw(outQuery))
  saveFile('queryImage.jpg', (outQuery).toBuffer('image/jpeg'))
  console.log('done, saved results to out/queryImage.jpg')
}


run()  


