import { useState, useRef, useCallback } from "react";
import api from "../services/api";
import {
  Prediction,
  FeedbackPayload,
  IAWAFeatureResult,
  FeatureSpeciesSupport,
} from "../types";
import { isTiff, tiffFileToDataUrl } from "../utils/tiffUtils";

export function useIdentify() {
  const [file, setFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [results, setResults] = useState<Prediction[]>([]);
  const [features, setFeatures] = useState<IAWAFeatureResult[]>([]);
  const [featureSupport, setFeatureSupport] = useState<FeatureSpeciesSupport>(
    {},
  );
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

  const handleFileSelect = async (selectedFile: File) => {
    revokePreview();
    setFile(selectedFile);
    setResults([]);
    setFeatures([]);
    setFeatureSupport({});
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";

    if (isTiff(selectedFile)) {
      try {
        const dataUrl = await tiffFileToDataUrl(selectedFile);
        // Data URLs are plain strings and don't need to be revoked via
        // URL.revokeObjectURL(), so previewUrlRef is intentionally left null.
        previewUrlRef.current = null;
        setImagePreview(dataUrl);
      } catch {
        setError("Failed to render TIFF image preview.");
      }
    } else {
      const url = URL.createObjectURL(selectedFile);
      previewUrlRef.current = url;
      setImagePreview(url);
    }
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
      setFeatures(resp.data.features || []);
      setFeatureSupport(resp.data.feature_species_support || {});
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
      setFeatures([]);
      setFeatureSupport({});
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
    features,
    featureSupport,
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
