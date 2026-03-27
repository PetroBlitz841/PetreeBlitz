import {
  Card,
  CardMedia,
  CardContent,
  Stack,
  Typography,
  Chip,
  Box,
  Skeleton,
} from "@mui/material";
import { Info } from "@mui/icons-material";
import { Image } from "../../types";
import { useTiffAwareSrc } from "../../hooks/useTiffAwareSrc";

interface ImageCardProps {
  image: Image;
  onClick?: () => void;
}

const getConfidenceColor = (confidence: number) => {
  if (confidence > 0.8) return "success";
  if (confidence > 0.5) return "warning";
  return "error";
};

export default function ImageCard({ image, onClick }: ImageCardProps) {
  const top = image.predictions[0];
  const rawSrc = image.image_url;
  const resolvedSrc = useTiffAwareSrc(rawSrc);

  return (
    <Card
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        transition: "transform 0.2s, box-shadow 0.2s",
        cursor: onClick ? "pointer" : "default",
        "&:hover": onClick
          ? { transform: "translateY(-4px)", boxShadow: 3 }
          : undefined,
      }}
      onClick={onClick}
    >
      {resolvedSrc ? (
        <CardMedia
          component="img"
          height="250"
          image={resolvedSrc}
          alt="Tree sample"
          sx={{ objectFit: "cover" }}
        />
      ) : (
        <Skeleton variant="rectangular" height={250} />
      )}
      <CardContent sx={{ flex: 1 }}>
        {top && (
          <Stack spacing={1} sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Top Prediction: {top.label}
            </Typography>
            <Typography variant="h6" fontWeight="bold">
              {image.sample_id}
            </Typography>
            <Chip
              label={
                image.feedback?.correct_label
                  ? "Corrected"
                  : `${(top.confidence * 100).toFixed(1)}%`
              }
              color={
                image.feedback?.correct_label
                  ? "default"
                  : (getConfidenceColor(top.confidence) as
                      | "success"
                      | "warning"
                      | "error")
              }
              size="small"
            />
          </Stack>
        )}
        {image.feedback && (
          <Box sx={{ pt: 1, borderTop: 1, borderColor: "grey.200" }}>
            <Stack direction="row" alignItems="center" spacing={1}>
              <Info fontSize="small" color="info" />
              <Typography variant="caption" color="text.secondary">
                {image.feedback.was_correct
                  ? "Correct prediction"
                  : `Corrected to: ${image.feedback.correct_label}`}
              </Typography>
            </Stack>
          </Box>
        )}
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ mt: 1, display: "block" }}
        >
          {new Date(image.timestamp).toLocaleString()}
        </Typography>
      </CardContent>
    </Card>
  );
}
