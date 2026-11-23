// Route /request-log — analysts review recent API requests (endpoint, time, status).

import RequestLogClient from "../ui/governance/_components/RequestLogClient";

export const metadata = {
  title: "Request Log | NBA Play Ranker",
};

export default function RequestLogPage() {
  return <RequestLogClient />;
}
