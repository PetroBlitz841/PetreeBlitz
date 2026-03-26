import { useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Box,
  Chip,
  Stack,
  Divider,
  Alert,
} from "@mui/material";
import { AddCircleOutline } from "@mui/icons-material";
import { FeatureCorrection } from "../types";

interface FeedbackDialogProps {
  open: boolean;
  loading?: boolean;
  featureCorrections?: FeatureCorrection[];
  onCancel: () => void;
  onSubmit: (speciesName: string) => void;
}

export default function FeedbackDialog({
  open,
  loading = false,
  featureCorrections = [],
  onCancel,
  onSubmit,
}: FeedbackDialogProps) {
  const [label, setLabel] = useState("");

  const handleClose = () => {
    setLabel("");
    onCancel();
  };

  const handleSubmit = () => {
    const name = label.trim();
    if (!name) return;
    onSubmit(name);
    setLabel("");
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Stack direction="row" alignItems="center" spacing={1}>
          <AddCircleOutline color="primary" />
          <span>Add New Tree Species</span>
        </Stack>
      </DialogTitle>
      <DialogContent sx={{ pt: 2 }}>
        <Typography variant="body2" sx={{ mb: 2 }}>
          None of the predicted species match the uploaded sample? Enter the
          correct species name to register it and teach the model.
        </Typography>
        <TextField
          autoFocus
          fullWidth
          label="Species name"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder="e.g., Byrsonima coriaceae"
          disabled={loading}
        />

        {/* Feature correction summary */}
        {featureCorrections.length > 0 && (
          <Box mt={2.5}>
            <Divider sx={{ mb: 1.5 }} />
            <Typography variant="subtitle2" gutterBottom>
              Feature corrections to include ({featureCorrections.length})
            </Typography>
            <Alert severity="info" sx={{ mb: 1, py: 0.5 }}>
              Your feature corrections from the breakdown panel will be sent
              together with this registration to guide model learning.
            </Alert>
            <Stack direction="row" flexWrap="wrap" gap={0.75}>
              {featureCorrections.map((fc) => (
                <Chip
                  key={fc.feature_id}
                  label={`Feature #${fc.feature_id} · ${fc.importance_weight}×`}
                  size="small"
                  color="warning"
                  variant="outlined"
                  sx={{ fontSize: "0.68rem" }}
                />
              ))}
            </Stack>
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={loading || !label.trim()}
          startIcon={
            loading ? <CircularProgress size={20} /> : <AddCircleOutline />
          }
        >
          {loading ? "Registering..." : "Register Species"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
