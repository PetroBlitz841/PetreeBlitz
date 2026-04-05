import {
  Box,
  Typography,
  Stack,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Divider,
} from "@mui/material";
import Grid from "@mui/material/Grid";
import SyncIcon from "@mui/icons-material/Sync";
import CloudDoneIcon from "@mui/icons-material/CloudDone";
import ScienceIcon from "@mui/icons-material/Science";
import { FederatedStats } from "../../types";

const LAB_COLORS = ["#2e7d32", "#1565c0", "#e65100", "#6a1b9a"];

interface FederatedSectionProps {
  federated: FederatedStats;
}

export default function FederatedSection({ federated }: FederatedSectionProps) {
  const totalContributions = federated.labs.reduce(
    (s, l) => s + l.identifications,
    0,
  );

  return (
    <Box>
      <Typography variant="h6" fontWeight={700} gutterBottom>
        Federated Learning Network
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={2}>
        Collaborative model training across research labs — only model updates
        are shared, raw data stays on each lab's devices
      </Typography>

      {/* Model version ribbon */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        sx={{
          mb: 3,
          p: 1.5,
          borderRadius: 2,
          bgcolor: "grey.100",
        }}
      >
        <CloudDoneIcon color="success" />
        <Stack spacing={0}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Typography variant="subtitle2" fontWeight={700}>
              Model {federated.model_version}
            </Typography>
            <Chip
              label={federated.version_hash}
              size="small"
              variant="outlined"
              sx={{ fontFamily: "monospace", fontSize: "0.7rem", height: 20 }}
            />
          </Stack>
          <Typography variant="caption" color="text.secondary">
            {federated.total_sync_updates} federated updates synced across{" "}
            {federated.labs.length} labs
          </Typography>
        </Stack>
        <Box flex={1} />
        <SyncIcon
          color="action"
          sx={{
            animation: "spin 4s linear infinite",
            "@keyframes spin": {
              "0%": { transform: "rotate(0deg)" },
              "100%": { transform: "rotate(360deg)" },
            },
          }}
        />
      </Stack>

      {/* Lab cards */}
      <Grid container spacing={2}>
        {federated.labs.map((lab, i) => {
          const color = LAB_COLORS[i % LAB_COLORS.length];
          const ratio =
            totalContributions > 0
              ? lab.identifications / totalContributions
              : 0;

          return (
            <Grid size={{ xs: 12, sm: 6, md: 3 }} key={lab.name}>
              <Card
                variant="outlined"
                sx={{
                  height: "100%",
                  borderTop: `3px solid ${color}`,
                  borderRadius: 2,
                }}
              >
                <CardContent sx={{ py: 2 }}>
                  <Stack spacing={1.5}>
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <ScienceIcon sx={{ color, fontSize: 20 }} />
                      <Typography variant="subtitle2" fontWeight={700}>
                        {lab.name}
                      </Typography>
                    </Stack>

                    <Chip
                      label={lab.region}
                      size="small"
                      variant="outlined"
                      sx={{
                        alignSelf: "flex-start",
                        fontSize: "0.7rem",
                        height: 20,
                      }}
                    />

                    <Divider />

                    <Stack spacing={0.5}>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">
                          Identifications
                        </Typography>
                        <Typography variant="caption" fontWeight={600}>
                          {lab.identifications}
                        </Typography>
                      </Stack>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">
                          Feedback
                        </Typography>
                        <Typography variant="caption" fontWeight={600}>
                          {lab.feedback}
                        </Typography>
                      </Stack>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">
                          Feature corrections
                        </Typography>
                        <Typography variant="caption" fontWeight={600}>
                          {lab.feature_corrections}
                        </Typography>
                      </Stack>
                    </Stack>

                    <Box>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        display="block"
                        mb={0.5}
                      >
                        Contribution share
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={ratio * 100}
                        sx={{
                          height: 6,
                          borderRadius: 3,
                          bgcolor: "grey.200",
                          "& .MuiLinearProgress-bar": {
                            bgcolor: color,
                          },
                        }}
                      />
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        display="block"
                        mt={0.25}
                        textAlign="right"
                      >
                        {(ratio * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
}
