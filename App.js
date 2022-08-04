import React, { useState, useEffect } from "react";
import { StatusBar } from "expo-status-bar";
import { LogBox, StyleSheet, Text, View, Button, Image } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { decodeJpeg, bundleResourceIO } from "@tensorflow/tfjs-react-native";
import * as ImagePicker from "expo-image-picker";
import * as MediaLibrary from "expo-media-library";
import * as FileSystem from "expo-file-system";

export default function App() {
  const [model, setModel] = useState(null);
  const [imagePath, setImagePath] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const class_name = [
    "black cutworm",
    "grain spreader thrips",
    "grub",
    "large cutworm",
    "mole cricket",
    "rice leafhopper",
    "rice shell pest",
    "rice water weevil",
    "white margined moth",
    "wireworm",
  ];

  useEffect(() => {
    async function loadModel() {
      await tf.ready();
      tf.setBackend("cpu");
      const modelJson = require("./assets/model/model.json");
      const modelWeight = require("./assets/model/group1-shard.bin");
      const pestClassifier = await tf.loadGraphModel(
        bundleResourceIO(modelJson, modelWeight)
      );
      setModel(pestClassifier);
    }
    loadModel();
  }, []);

  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });
    console.log(result);
    if (!result.cancelled) {
      setImagePath(result.uri);
    }
  };

  const openCamera = async () => {
    // Ask the user for the permission to access the camera
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (permissionResult.granted === false) {
      alert("You've refused to allow this appp to access your camera!");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
    });
    console.log(result);
    if (!result.cancelled) {
      setImagePath(result.uri);
      console.log(result.uri);
    }
  };

  const saveToPhone = async (imagePath) => {
    const permission = await MediaLibrary.requestPermissionsAsync();
    if (permission.granted) {
      try {
        const asset = await MediaLibrary.createAssetAsync(imagePath);
        MediaLibrary.createAlbumAsync("PestClassification", asset, false)
          .then(() => {
            console.log("File Saved Successfully!");
          })
          .catch(() => {
            console.log("Error In Saving File!");
          });
      } catch (error) {
        console.log(error);
      }
    } else {
      console.log("Need Storage permission to save file");
    }
  };

  const classifyImage = async (model, imagePath) => {
    const imgB64 = await FileSystem.readAsStringAsync(imagePath, {
      encoding: FileSystem.EncodingType.Base64,
    });
    const imgBuffer = tf.util.encodeString(imgB64, "base64").buffer;
    const imageData = new Uint8Array(imgBuffer);
    const IMG_SIZE = 224;
    const imageTensor = decodeJpeg(imageData)
      .expandDims()
      .resizeBilinear([IMG_SIZE, IMG_SIZE])
      .reshape([1, IMG_SIZE, IMG_SIZE, 3]);
    console.log(imageTensor);

    const pestClassifier = model;
    const result = pestClassifier.predict(imageTensor);
    const prediction = result.dataSync();
    const confidence = Math.max(...prediction);
    const index = prediction.indexOf(confidence);
    console.log("Prediction:", prediction);
    setPrediction(index);
    setConfidence(confidence);
  };

  return (
    <View style={styles.container}>
      <Text>
        Model ready?{" "}
        {model == null ? <Text>Loading Model...</Text> : <Text>Yes</Text>}
      </Text>
      <StatusBar style="auto" />

      <View>
        <Button title="Pick Image" onPress={pickImage} />
        <Button title="Open Camera" onPress={openCamera} />
      </View>

      <View style={{ padding: 30 }}>
        {imagePath && (
          <Image
            source={{ uri: imagePath }}
            style={{ width: 224, height: 224 }}
          />
        )}
      </View>

      <View>
        <Button
          title="Classify Image"
          onPress={() => {
            saveToPhone(imagePath);
            classifyImage(model, imagePath);
          }}
        />
      </View>

      <View>
        <Text>
          Prediction:{" "}
          {prediction == null ? (
            <Text>...</Text>
          ) : (
            <Text>
              {(confidence * 100).toFixed(2)}% {class_name[prediction]}
            </Text>
          )}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
});

LogBox.ignoreAllLogs(true);
