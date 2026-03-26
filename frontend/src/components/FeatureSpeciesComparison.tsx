import { useState } from "react";
import {
  Box,
  Stack,
  Typography,
  Chip,
  Collapse,
  IconButton,
  Divider,
  Tooltip,
  LinearProgress,
  Button,
} from "@mui/material";
import {
  CheckCircle,
  Cancel,
  ExpandMore,
  ExpandLess,
  AddCircleOutline,
} from "@mui/icons-material";
import { IAWAFeatureResult, FeatureSpeciesSupport, Prediction } from "../types";

interface FeatureSpeciesComparisonProps {
  predictions: Prediction[];
  featureSupport: FeatureSpeciesSupport;
  allFeatures: IAWAFeatureResult[];
  feedbackLoading?: boolean;
  onCorrect: (label: string) => void;
  onNewSpecies: () => void;
}

function matchColor(ratio: number): "success" | "warning" | "error" {
  return ratio >= 0.7 ? "success" : ratio >= 0.45 ? "warning" : "error";
}

function confidenceColor(c: number): "success" | "warning" | "error" {
  return c >= 0.8 ? "success" : c >= 0.5 ? "warning" : "error";
}

interface SpeciesCardProps {
  index: number;
  prediction: Prediction;
  support: { matched: number[]; mismatched: number[] } | undefined;
  allFeatures: IAWAFeatureResult[];
  feedbackLoading: boolean;
  onCorrect: (label: string) => void;
}

function SpeciesCard({
  index,
  prediction,
  support,
  allFeatures,
  feedbackLoading,
  onCorrect,
}: SpeciesCardProps) {
  const [expanded, setExpanded] = useState(false);

  const featureMap = new Map(allFeatures.map((f) => [f.id, f]));
  const matched = (support?.matched ?? [])
    .map((id) => featureMap.get(id))
    .filter(Boolean) as IAWAFeatureResult[];
  const mismatched = (support?.mismatched ?? [])
    .map((id) => featureMap.get(id))
    .filter(Boolean) as IAWAFeatureResult[];
  const total = matched.length + mismatched.length;
  const ratio = total > 0 ? matched.length / total : 0;
  const hasFeatures = total > 0;

  return (
    <Box
      sx={{
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 2,
        p: 2,
      }}
    >
      {/* Header: rank + label + confidence + action */}
      <Stack direction="row" alignItems="center" spacing={1}>
        <Chip
          label={`#${index + 1}`}
          size="small"
          variant="outlined"
          sx={{ fontWeight: 700, minWidth: 34 }}
        />
        <Stack direction="column" flex={1} spacing={0.25}>
          <Stack direction="row" alignItems="center" spacing={0.75}>
            <Typography variant="subtitle1" fontWeight={700}>
              {prediction.label.replace(/_/g, " ")}
            </Typography>
            <Chip
              label={`${(prediction.confidence * 100).toFixed(1)}%`}
              size="small"
              color={confidenceColor(prediction.confidence)}
              sx={{ height: 20, fontSize: "0.7rem" }}
            />
          </Stack>

          {/* Feature match progress bar */}
          {hasFeatures && (
            <Stack direction="row" alignItems="center" spacing={1}>
              <LinearProgress
                variant="determinate"
                value={ratio * 100}
                color={matchColor(ratio)}
                sx={{ flex: 1, height: 6, borderRadius: 3 }}
              />
              <Typography
                variant="caption"
                color="text.secondary"
                minWidth={70}
              >
                {matched.length}/{total} features
              </Typography>
            </Stack>
          )}
        </Stack>

        {/* Actions */}
        <Stack direction="row" alignItems="center" spacing={0.5}>
          <Button
            variant="outlined"
            color="success"
            size="small"
            startIcon={<CheckCircle />}
            onClick={() => onCorrect(prediction.label)}
            disabled={feedbackLoading}
            sx={{ whiteSpace: "nowrap" }}
          >
            This is correct
          </Button>
          {hasFeatures && (
            <Tooltip title={expanded ? "Hide features" : "Show features"}>
              <IconButton size="small" onClick={() => setExpanded((v) => !v)}>
                {expanded ? (
                  <ExpandLess fontSize="small" />
                ) : (
                  <ExpandMore fontSize="small" />
                )}
              </IconButton>
            </Tooltip>
          )}
        </Stack>
      </Stack>

      {/* Expanded feature lists */}
      <Collapse in={expanded}>
        <Box mt={1.5}>
          {matched.length > 0 && (
            <Box mb={1}>
              <Typography
                variant="caption"
                color="success.main"
                fontWeight={600}
                display="block"
                mb={0.5}
              >
                Matched features ({matched.length})
              </Typography>
              <Stack spacing={0.25}>
                {matched.map((f) => (
                  <Stack
                    key={f.id}
                    direction="row"
                    alignItems="center"
                    spacing={0.75}
                  >
                    <CheckCircle
                      fontSize="inherit"
                      color="success"
                      sx={{ fontSize: 13 }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      {f.name}
                    </Typography>
                  </Stack>
                ))}
              </Stack>
            </Box>
          )}
          {mismatched.length > 0 && (
            <Box>
              <Divider sx={{ mb: 1 }} />
              <Typography
                variant="caption"
                color="error.main"
                fontWeight={600}
                display="block"
                mb={0.5}
              >
                Mismatched features ({mismatched.length})
              </Typography>
              <Stack spacing={0.25}>
                {mismatched.map((f) => (
                  <Stack
                    key={f.id}
                    direction="row"
                    alignItems="center"
                    spacing={0.75}
                  >
                    <Cancel
                      fontSize="inherit"
                      color="error"
                      sx={{ fontSize: 13 }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      {f.name}
                    </Typography>
                  </Stack>
                ))}
              </Stack>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}

export default function FeatureSpeciesComparison({
  predictions,
  featureSupport,
  allFeatures,
  feedbackLoading = false,
  onCorrect,
  onNewSpecies,
}: FeatureSpeciesComparisonProps) {
  if (predictions.length === 0) return null;

  const top3 = predictions.slice(0, 3);

  return (
    <Box>
      <Typography variant="h5" gutterBottom fontWeight={700}>
        Identification Results
      </Typography>
      {allFeatures.length > 0 && (
        <Typography variant="body2" color="text.secondary" mb={1.5}>
          Ranked by confidence · expand a row to see how IAWA features align
        </Typography>
      )}
      <Stack spacing={1.5}>
        {top3.map((pred, i) => (
          <SpeciesCard
            key={pred.label}
            index={i}
            prediction={pred}
            support={featureSupport[pred.label]}
            allFeatures={allFeatures}
            feedbackLoading={feedbackLoading}
            onCorrect={onCorrect}
          />
        ))}
      </Stack>

      <Divider sx={{ mt: 2, mb: 1 }} />
      <Stack direction="row" justifyContent="center">
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
    </Box>
  );
}
