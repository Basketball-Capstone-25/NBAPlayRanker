// Service Layer — late-game (clutch) report (SCRUM-629).
export {
  clutchFilename,
  clutchReportUrl,
  fetchClutchReport,
  ClutchAccessError,
  type ClutchFormat,
} from "../infrastructure/clutch-report";
export { authenticatedDownload } from "../infrastructure/api-client";
