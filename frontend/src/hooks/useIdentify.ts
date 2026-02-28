import { useState, useRef, useCallback } from "react";
import api from "../services/api";
import { Prediction, FeedbackPayload } from "../types";

export function useIdentify() {
  const [file, setFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [results, setResults] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);

  const previewUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const revokePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
  }, []);

  const clearImage = useCallback(() => {
    revokePreview();
    setImagePreview(null);
    setFile(null);
  }, [revokePreview]);

  const handleFileSelect = (selectedFile: File) => {
    revokePreview();
    const url = URL.createObjectURL(selectedFile);
    previewUrlRef.current = url;
    setFile(selectedFile);
    setImagePreview(url);
    setResults([]);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const identify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await api.post("/identify", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSampleId(resp.data.sample_id);
      setResults(resp.data.predictions || []);
    } catch (err) {
      console.error(err);
      setError("Failed to identify tree species. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (payload: FeedbackPayload) => {
    setLoading(true);
    setError(null);
    try {
      await api.post("/feedback", payload);
      clearImage();
      setResults([]);
      setSampleId(null);
    } catch (err) {
      console.error(err);
      setError("Failed to submit feedback. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return {
    file,
    imagePreview,
    results,
    loading,
    error,
    sampleId,
    fileInputRef,
    setError,
    handleFileSelect,
    identify,
    sendFeedback,
    clearImage,
  };
}
