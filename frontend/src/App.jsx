import { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);

  const [gleasonScore, setGleasonScore] = useState("");
  const [mask, setMask] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));

    // Reset previous prediction
    setMask(null);
    setGleasonScore("");
    setError("");
  };

  const handleDiagnose = async () => {
    if (!image) {
      setError("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed.");
      }

      const data = await response.json();

      setGleasonScore(data.gleason_score);
      setMask(`data:image/png;base64,${data.mask}`);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>GleasonAI</h1>

      <p className="subtitle">
        AI-assisted Gleason grading from prostate histopathology images.
      </p>

      <div className="upload-section">
        <input type="file" accept="image/*" onChange={handleFileChange} />

        <button onClick={handleDiagnose} disabled={loading}>
          {loading ? "Diagnosing..." : "Diagnose"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {(preview || mask) && (
        <div className="images">
          <div className="card">
            <h2>Original Image</h2>

            {preview && <img src={preview} alt="Original" />}
          </div>

          <div className="card">
            <h2>Segmentation</h2>

            {mask && <img src={mask} alt="Prediction" />}
          </div>
        </div>
      )}

      {gleasonScore && (
        <div className="result">
          <h2>Diagnosis</h2>

          <p>Gleason Score</p>

          <h1>{gleasonScore}</h1>
        </div>
      )}
    </div>
  );
}
