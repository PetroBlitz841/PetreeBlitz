import { Stack, Grid, Typography, Button, Divider } from "@mui/material";
import { AddCircleOutline } from "@mui/icons-material";
import {
  Prediction,
  IAWAFeatureResult,
  FeatureSpeciesSupport,
} from "../types";
import PredictionCard from "./PredictionCard";

interface ResultsListProps {
  results: Prediction[];
  feedbackLoading?: boolean;
  featureSupport?: FeatureSpeciesSupport;
  allFeatures?: IAWAFeatureResult[];
  onCorrect: (label: string) => void;
  onNewSpecies: () => void;
}

export default function ResultsList({
  results,
  feedbackLoading = false,
  featureSupport,
  allFeatures,
  onCorrect,
  onNewSpecies,
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
              featureSupport={featureSupport}
              allFeatures={allFeatures}
              onCorrect={onCorrect}
            />
          </Grid>
        ))}
      </Grid>

      <Divider />

      {/* New species action */}
      <Stack direction="row" alignItems="center" justifyContent="center">
        <Button
          variant="text"
          color="primary"
          size="small"
          startIcon={<AddCircleOutline />}
          onClick={onNewSpecies}
          disabled={feedbackLoading}
        >
          None of these match — add as new species
        </Button>
      </Stack>
    </Stack>
  );
}

