# Dataset sources

## Play-type data

The play-type CSV comes from NBA Stats / Synergy through hoopR. The project has player rows for 2019–20 through 2024–25. Its original download date was not recorded, so the app shows it as unknown. A file hash identifies the exact copy used by the service.

The teammate's shared dataset and source discussion were found in the project messages. The earlier Synergy inquiry did not provide a data license. Private messages and email are not included here.

## Shot data

The repaired shot data comes from [SportsDataverse's ESPN NBA play-by-play release](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_pbp), loaded through the source documented by hoopR. The six season files were downloaded on October 25, 2026. Their URLs, sizes and hashes are in [source-manifest.json](../shot-repair/source-manifest.json).

Attribution: ESPN play-by-play data, distributed by SportsDataverse. The [SportsDataverse data README](https://github.com/sportsdataverse/sportsdataverse-data) identifies its data license as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). NBA Play Ranker filters field-goal attempts, maps event team IDs to game participants, calculates score margin before each shot, and builds shot summaries. These are project transformations, not unmodified provider files. No provider endorsement is implied.

The raw files stay outside the repository. The downloader checks the pinned hashes and refuses a changed release asset. Older handoff packages retain their historical data and should not be used to validate the repaired shot features.

## Permission follow-up — BSR-159

The provider's data notice and hoopR's software license are recorded separately. Neither is proof of a direct NBA/Synergy permission grant. [NBA terms](https://www.nba.com/termsofuse), including section 9, require a separate review before broader sharing. ESPN's [general terms](https://disneytermsofuse.com/english/) also need to be considered for the upstream content.

Gurk owns the follow-up: confirm whether the intended hosted academic demo, reports and source/data handoff are covered. Keep the raw files out of the submission ZIP while this is unresolved. If a rights holder requires a change, replace that input or use clearly labelled synthetic fixtures for the demo. Do not describe public download access as unrestricted redistribution permission.

The source search covered relevant user files, archive indexes, Git history, connected email and the teammate chats in Discord and Teams. It found no original raw play-by-play copy. The new provider download has its own retrieval date and hashes; it is not presented as the original historical download.
