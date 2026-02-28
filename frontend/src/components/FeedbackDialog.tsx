import { useState } from "react";
import { Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  TextField,
  Button,
  CircularProgress,
} from "@mui/material";

interface FeedbackDialogProps {
  open: boolean;
  predictedLabel?: string;
  loading?: boolean;
  onCancel: () => void;
  onSubmit: (correctLabel?: string) => void;
}

export default function FeedbackDialog({
  open,
  predictedLabel,
  loading = false,
  onCancel,
  onSubmit,
}: FeedbackDialogProps) {
  const [label, setLabel] = useState("");

  const handleClose = () => {
    setLabel("");
    onCancel();
  };

  const handleSubmit = () => {
    onSubmit(label.trim() || undefined);
    setLabel("");
  };


  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Correct Species Label</DialogTitle>
      <DialogContent sx={{ pt: 2 }}>
        <Typography variant="body2" sx={{ mb: 2 }}>
          The model predicted: <strong>{predictedLabel}</strong>
        </Typography>
        <Typography variant="body2" sx={{ mb: 2 }}>
          What is the correct species name?
        </Typography>
        <TextField
          autoFocus
          fullWidth
          label="Correct species label"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="e.g., Oak, Pine, Birch"
          disabled={loading}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : null}
        >
          {loading ? "Submitting..." : "Submit Feedback"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
