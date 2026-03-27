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
  Skeleton,
} from "@mui/material";
import { Image } from "../../types";
import { useTiffAwareSrc } from "../../hooks/useTiffAwareSrc";

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

function DialogImage({ image }: { image: Image }) {
  const rawSrc = image.image_url;
  const resolvedSrc = useTiffAwareSrc(rawSrc);

  if (!resolvedSrc) return <Skeleton variant="rectangular" height={400} />;
  return (
    <Box
      component="img"
      src={resolvedSrc}
      alt="Tree sample"
      sx={{ width: "100%", maxHeight: 400, objectFit: "contain" }}
    />
  );
}

export default function ImageDetailsDialog({
  open,
  image,
  onClose,
}: ImageDetailsDialogProps) {
  if (!image) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Image Details</DialogTitle>
      <DialogContent sx={{ pt: 2 }}>
        <Stack spacing={3}>
          <DialogImage image={image} />

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
                  color={
                    getConfidenceColor(pred.confidence) as
                      | "success"
                      | "warning"
                      | "error"
                  }
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
                <strong>Timestamp:</strong>{" "}
                {new Date(image.timestamp).toLocaleString()}
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
