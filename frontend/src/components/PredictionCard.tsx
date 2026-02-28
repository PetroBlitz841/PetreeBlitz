import {
  Card,
  CardContent,
  Typography,
  Chip,
  Stack,
  Button,
} from "@mui/material";
import { CheckCircle, Cancel } from "@mui/icons-material";
import { Prediction } from "../types";

interface PredictionCardProps {
  prediction: Prediction;
  index: number;
  feedbackLoading?: boolean;
  onCorrect: (label: string) => void;
  onWrong: (label: string) => void;
}

export default function PredictionCard({
  prediction,
  index,
  feedbackLoading = false,
  onCorrect,
  onWrong,
}: PredictionCardProps) {
  const color: "success" | "warning" | "error" =
    prediction.confidence > 0.8
      ? "success"
      : prediction.confidence > 0.5
      ? "warning"
      : "error";

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" fontWeight="bold">
          #{index + 1} {prediction.label}
        </Typography>
        <Chip
          label={`${(prediction.confidence * 100).toFixed(1)}%`}
          color={color}
          sx={{ mb: 1 }}
        />
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            color="success"
            size="small"
            startIcon={<CheckCircle />}
            onClick={() => onCorrect(prediction.label)}
            disabled={feedbackLoading}
          >
            Correct
          </Button>
          <Button
            variant="outlined"
            color="error"
            size="small"
            startIcon={<Cancel />}
            onClick={() => onWrong(prediction.label)}
            disabled={feedbackLoading}
          >
            Wrong
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
}
