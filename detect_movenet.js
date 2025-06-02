// This script uses TensorFlow.js and the MoveNet model to perform pose detection
// on an image, excluding certain keypoints and calculating weighted averages
// for the remaining keypoints over multiple trials.
// To run: node detect_movenet.js
// Requires Node.js, TensorFlow.js, and the MoveNet model.
// Make sure to install the required packages:
// npm install @tensorflow/tfjs-node @tensorflow-models/pose-detection
// Make sure to have an image at 'assets/startFrame copy.png' or change the path accordingly.
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const poseDetection = require('@tensorflow-models/pose-detection');

const EXCLUDED_KEYPOINTS = new Set([
  'right_ankle', 'left_ankle',
  'right_knee', 'left_knee',
  'right_hip', 'left_hip',
  'right_wrist', 'left_wrist',
  'right_elbow', 'left_elbow',
]);

async function weightedPoseDetection(imagePath, numTrials = 50) {
  // Load model once
  const detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    }
  );

  // Decode image once
  const imageBuffer = fs.readFileSync(imagePath);
  const imageTensor = tf.node.decodeImage(imageBuffer, 3);

  const weightedSums = new Map();

  for (let i = 0; i < numTrials; i++) {
    const poses = await detector.estimatePoses(imageTensor);
    if (poses.length === 0) continue;

    poses[0].keypoints.forEach(kp => {
      if (EXCLUDED_KEYPOINTS.has(kp.name)) return;
      if (!weightedSums.has(kp.name)) {
        weightedSums.set(kp.name, { x: 0, y: 0, totalScore: 0 });
      }
      const entry = weightedSums.get(kp.name);
      entry.x += kp.x * kp.score;
      entry.y += kp.y * kp.score;
      entry.totalScore += kp.score;
    });

    if (i % 10 === 0) await tf.nextFrame();
  }

  imageTensor.dispose();

  // Output weighted averages
  for (const [name, { x, y, totalScore }] of weightedSums.entries()) {
    if (totalScore === 0) continue;
    const avgX = x / totalScore;
    const avgY = y / totalScore;
    const avgScore = totalScore / numTrials;
    console.log(`${name.padEnd(16)} x: ${avgX.toFixed(1)}, y: ${avgY.toFixed(1)}, score: ${avgScore.toFixed(3)}`);
  }
}

weightedPoseDetection('assets/startFrame copy.png', 50);
