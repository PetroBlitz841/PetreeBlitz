import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Stack,
  Chip,
} from "@mui/material";
import { Image } from "../types";

interface ImageDetailsDialogProps {
  open: boolean;
  image: Image | null;
  onClose: () => void;
}

const getConfidenceColor = (confidence: number) => {
  if (confidence > 0.8) return "success";
  if (confidence > 0.5) return "warning";
  return "error";
};

export default function ImageDetailsDialog({ open, image, onClose }: ImageDetailsDialogProps) {
  if (!image) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Image Details</DialogTitle>
      <DialogContent sx={{ pt: 2 }}>
        <Stack spacing={3}>
          <Box
            component="img"
            src={image.image_url.startsWith("http") ? image.image_url : `http://localhost:8000${image.image_url}`}
            alt="Tree sample"
            sx={{ width: "100%", maxHeight: 400, objectFit: "contain" }}
          />

          <Stack spacing={1}>
            <Typography variant="subtitle1" fontWeight="bold">
              All Predictions
            </Typography>
            {image.predictions.map((pred, idx) => (
              <Box
                key={idx}
                sx={{
                  p: 2,
                  backgroundColor: "grey.50",
                  borderRadius: 1,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <Typography>
                  {idx + 1}. {pred.label}
                </Typography>
                <Chip
                  label={`${(pred.confidence * 100).toFixed(1)}%`}
                  color={getConfidenceColor(pred.confidence) as "success" | "warning" | "error"}
                />
              </Box>
            ))}
          </Stack>

          {image.feedback && (
            <Stack spacing={1}>
              <Typography variant="subtitle1" fontWeight="bold">
                Feedback
              </Typography>
              <Box sx={{ p: 2, backgroundColor: "blue.50", borderRadius: 1 }}>
                <Typography variant="body2">
                  {image.feedback.was_correct
                    ? "✓ Prediction was marked as correct"
                    : `✓ Correction provided: ${image.feedback.correct_label}`}
                </Typography>
              </Box>
            </Stack>
          )}

          <Stack spacing={1}>
            <Typography variant="subtitle1" fontWeight="bold">
              Metadata
            </Typography>
            <Box sx={{ p: 2, backgroundColor: "grey.50", borderRadius: 1 }}>
              <Typography variant="caption" display="block">
                <strong>Sample ID:</strong> {image.sample_id}
              </Typography>
              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                <strong>Timestamp:</strong> {new Date(image.timestamp).toLocaleString()}
              </Typography>
            </Box>
          </Stack>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
