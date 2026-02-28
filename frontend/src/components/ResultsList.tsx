import { Stack, Grid, Typography } from "@mui/material";
import { Prediction } from "../types";
import PredictionCard from "./PredictionCard";

interface ResultsListProps {
  results: Prediction[];
  feedbackLoading?: boolean;
  onCorrect: (label: string) => void;
  onWrong: (label: string) => void;
}

export default function ResultsList({
  results,
  feedbackLoading = false,
  onCorrect,
  onWrong,
}: ResultsListProps) {
  if (results.length === 0) return null;

  return (
    <Stack direction="column" spacing={2} sx={{ width: "100%" }}>
      <Typography variant="h5">Identification Results</Typography>
      <Grid container spacing={2}>
        {results.map((prediction, index) => (
          <Grid size={{ xs: 12, sm: 4 }} key={index}>
            <PredictionCard
              index={index}
              prediction={prediction}
              feedbackLoading={feedbackLoading}
              onCorrect={onCorrect}
              onWrong={onWrong}
            />
          </Grid>
        ))}
      </Grid>
    </Stack>
  );
}
