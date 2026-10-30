// Service Layer — Data Explorer play-by-play shots (SCRUM-627).
export {
  canSearchShots,
  fetchShotPage,
  shotExplorerUrl,
  shotExportFilename,
  shotExportUrl,
  ShotAccessError,
  ShotDataUnavailableError,
  SHOT_PAGE_SIZE,
  type ShotColumn,
  type ShotFilters,
  type ShotPage,
  type ShotResult,
  type ShotRow,
} from "../infrastructure/shot-explorer";
export { authenticatedDownload, fetchPbpMetaOptions, type PbpMetaOptions } from "../infrastructure/api-client";
