import {
  Card,
  CardContent,
  Box,
  Stack,
  Typography,
} from "@mui/material";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { Album } from "../types";

interface AlbumCardProps {
  album: Album;
  onClick?: () => void;
}

export default function AlbumCard({ album, onClick }: AlbumCardProps) {
  return (
    <Card
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        transition: "transform 0.2s, box-shadow 0.2s",
        cursor: onClick ? "pointer" : "default",
        "&:hover": onClick
          ? {
              transform: "translateY(-4px)",
              boxShadow: 3,
            }
          : undefined,
      }}
      onClick={onClick}
    >
      <Box
        sx={{
          height: 200,
          backgroundColor: "grey.200",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <FolderOpenIcon sx={{ fontSize: 60, color: "grey.400" }} />
      </Box>
      <CardContent sx={{ flex: 1 }}>
        <Stack
          direction="row"
          justifyContent="space-between"
          alignItems="flex-start"
          sx={{ mb: 2 }}
        >
          <Typography
            variant="h6"
            fontWeight="bold"
            sx={{
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              flex: 1,
            }}
          >
            {album.name}
          </Typography>
        </Stack>
        <Stack
          direction="row"
          justifyContent="space-between"
          alignItems="center"
          sx={{ pt: 1, borderTop: 1, borderColor: "grey.200" }}
        >
          <Typography variant="caption" color="text.secondary">
            <strong>{album.num_images}</strong> image{album.num_images !== 1 ? "s" : ""}
          </Typography>
          <ArrowForwardIcon fontSize="small" sx={{ color: "primary.main" }} />
        </Stack>
      </CardContent>
    </Card>
  );
}
