import { useState } from "react";
import axios from "axios";
import "./App.css";
import { FiUploadCloud } from "react-icons/fi";
import { HiCheckCircle } from "react-icons/hi";

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

    loadImage(file);
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

  const getGradeColor = (grade) => {
    switch (Number(grade)) {
      case 3:
        return "rgb(0, 0, 255)"; // Blue (Grade 3)
      case 4:
        return "rgb(255, 230, 0)"; // Yellow (Grade 4)
      case 5:
        return "rgb(255, 0, 0)"; // Red (Grade 5)
      case 1:
        return "rgb(0, 128, 0)"; // Green (Benign)
      default:
        return "#0f4c81"; // Default blue
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();

    const file = e.dataTransfer.files[0];

    if (!file) return;

    loadImage(file);
  };

  const loadImage = (file) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));

    setMask(null);
    setGleasonScore("");
    setError("");
  };

  return (
    <div className="app">
      <h1>GleasonAI</h1>

      <p className="subtitle">
        AI-assisted Gleason grading from prostate histopathology images.
      </p>

      <div className="upload-section">
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          hidden
          onChange={handleFileChange}
        />

        <div
          className="drop-zone"
          onClick={() => document.getElementById("file-upload").click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {!image ? (
            <>
              <div className="upload-icon">
                <FiUploadCloud />
              </div>

              <h2>Drag & Drop a Tissue Image</h2>

              <p>or click to browse your computer</p>
            </>
          ) : (
            <>
              <div className="upload-icon">
                <HiCheckCircle />
              </div>

              <h2>{image.name}</h2>

              <p>Ready for diagnosis</p>

              <span>Click here to choose another image</span>
            </>
          )}
        </div>

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
            <div className="legend">
              <div className="legend-item">
                <span className="color background"></span>
                <span>Background</span>
              </div>

              <div className="legend-item">
                <span className="color benign"></span>
                <span>Benign</span>
              </div>

              <div className="legend-item">
                <span className="color grade3"></span>
                <span>Grade 3</span>
              </div>

              <div className="legend-item">
                <span className="color grade4"></span>
                <span>Grade 4</span>
              </div>

              <div className="legend-item">
                <span className="color grade5"></span>
                <span>Grade 5</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {gleasonScore && (
        <div className="result">
          <h2>Diagnosis</h2>

          <div className="diagnosis-row">
            <span>Primary Pattern</span>

            <strong
              style={{
                color: getGradeColor(gleasonScore.split("=")[0].split("+")[0]),
              }}
            >
              {gleasonScore.split("=")[0].split("+")[0]}
            </strong>
          </div>

          <div className="diagnosis-row">
            <span>Secondary Pattern</span>

            <strong
              style={{
                color: getGradeColor(gleasonScore.split("=")[0].split("+")[1]),
              }}
            >
              {gleasonScore.split("=")[0].split("+")[1]}
            </strong>
          </div>

          <hr />

          <div className="diagnosis-row total">
            <span>Gleason Score</span>
            <strong>{gleasonScore}</strong>
          </div>
        </div>
      )}

      <footer className="disclaimer">
        <strong>Disclaimer:</strong> GleasonAI is a research prototype developed
        as part of my Master's dissertation at the University of Edinburgh. It
        is intended for educational and demonstration purposes only and must not
        be used for clinical diagnosis, medical decision-making, or patient
        care.
      </footer>
    </div>
  );
}
