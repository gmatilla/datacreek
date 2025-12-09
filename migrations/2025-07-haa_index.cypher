CREATE INDEX haa_pair IF NOT EXISTS
FOR ()-[r:SUGGESTED_HYPER_AA]-()
ON (r.startNodeId, r.endNodeId);
