import Grid from "@mui/material/Grid";
import ParkIcon from "@mui/icons-material/Park";
import PhotoCameraIcon from "@mui/icons-material/PhotoCamera";
import ThumbUpAltIcon from "@mui/icons-material/ThumbUpAlt";
import MemoryIcon from "@mui/icons-material/Memory";
import BiotechIcon from "@mui/icons-material/Biotech";
import RateReviewIcon from "@mui/icons-material/RateReview";
import StatCard from "./StatCard";
import { OverviewStats } from "../../types";

interface OverviewSectionProps {
  stats: OverviewStats;
  accuracy: number | null;
}

export default function OverviewSection({
  stats,
  accuracy,
}: OverviewSectionProps) {
  return (
    <Grid container spacing={2}>
      <Grid size={{ xs: 6, sm: 4, md: 2 }}>
        <StatCard
          icon={<ParkIcon />}
          label="Species Known"
          value={stats.total_species}
          gradient={["#2e7d32", "#4caf50"]}
        />
      </Grid>
      <Grid size={{ xs: 6, sm: 4, md: 2 }}>
        <StatCard
          icon={<PhotoCameraIcon />}
          label="Identifications"
          value={stats.total_identifications}
          gradient={["#1565c0", "#42a5f5"]}
        />
      </Grid>
      <Grid size={{ xs: 6, sm: 4, md: 2 }}>
        <StatCard
          icon={<RateReviewIcon />}
          label="Feedback"
          value={stats.total_feedback}
          gradient={["#6a1b9a", "#ab47bc"]}
        />
      </Grid>
      <Grid size={{ xs: 6, sm: 4, md: 2 }}>
        <StatCard
          icon={<ThumbUpAltIcon />}
          label="Accuracy"
          value={accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : "—"}
          gradient={["#e65100", "#ff9800"]}
        />
      </Grid>
      <Grid size={{ xs: 6, sm: 4, md: 2 }}>
        <StatCard
          icon={<MemoryIcon />}
          label="Learned Embeddings"
          value={stats.total_learned_embeddings}
          subtitle={`${stats.total_original_embeddings} original`}
          gradient={["#00838f", "#26c6da"]}
        />
      </Grid>
      <Grid size={{ xs: 6, sm: 4, md: 2 }}>
        <StatCard
          icon={<BiotechIcon />}
          label="Feature Corrections"
          value={stats.total_feature_corrections}
          gradient={["#ad1457", "#ec407a"]}
        />
      </Grid>
    </Grid>
  );
}
