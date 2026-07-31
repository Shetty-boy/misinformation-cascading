# Cluster Runbook

Operational procedures for the CASCADE2VEC Spark + HDFS cluster. Read this before
touching the cluster, especially before adding a new node.

## Current cluster

See `cluster/nodes.txt` for the live list. Master/NameNode: whichever host is set as
`MASTER_HOST` (default `laptop-a`) — same machine runs both roles for simplicity.

## Adding a new node

1. Append the new machine's Tailscale MagicDNS hostname to `cluster/nodes.txt`.
2. Ensure Tailscale + MagicDNS is set up on the new machine and it can ping every
   existing node by hostname.
3. Ensure the new machine has matching Hadoop/Spark versions installed
   (`spark-submit --version` must match across all nodes).
4. Set up passwordless SSH from the master to the new node.
5. Run `cluster/setup_node.sh worker` on the new machine.
6. From the master: `hdfs dfsadmin -refreshNodes` (if the cluster was already
   running) or restart `start-dfs.sh` (if bringing the whole cluster up fresh).
7. **Run `hdfs balancer` from the master.** New nodes do not automatically receive
   a share of existing data — blocks written before the node joined stay where they
   were. This step is required every time a node is added to an already-populated
   cluster, not optional.
8. Verify: HDFS UI (`http://<master>:9870`) shows the new DataNode as live. Spark UI
   (`http://<master>:8080`) shows the new executor.

## Replication factor vs. node count — known tradeoff

`dfs.replication` is set in `hdfs-site.xml`, currently `2`.

- **At 2 total nodes**: replication=2 means every block is stored on *both* nodes.
  This is functionally full replication — there is no valid placement where a block
  exists on only one machine, since 2 copies are required and only 2 machines exist.
  This is expected, not a misconfiguration. The "no single machine holds all the
  data" property does not hold at this cluster size.
- **At 3+ nodes with replication=2**: real partial placement kicks in. Different
  blocks land on different pairs of machines, so no single node holds 100% of the
  data. This is the point at which HDFS actually demonstrates chunked/distributed
  storage rather than dressed-up replication.
- **Action**: when reporting results with fewer than 3 nodes, state explicitly in
  any write-up that the storage-locality benefit is not yet in effect — only the
  compute-parallelism benefit is being measured at that cluster size.

## Fallback: if HDFS setup is blocking progress

Full replication via `scp` or a shared NFS mount is an acceptable fallback if HDFS
setup is consuming time that should go to the research track (feature engineering,
embedding, etc.). If this fallback is used, document it explicitly in the paper's
infrastructure section — it changes what the scalability results actually
demonstrate (compute scaling only, not storage-aware scheduling).

## Verifying the cluster is real, not just "didn't error"

Don't trust a job finishing without errors as proof the cluster is working. Check:

- Spark UI → Executors tab: confirm tasks are distributed across hosts, not all
  running on the driver's local cores.
- Spark UI → a running/completed job's task list: look for `NODE_LOCAL` task
  locality on multiple hosts (proof of locality-aware scheduling actually working),
  vs. `ANY`/`RACK_LOCAL` (data had to travel over the network).
- HDFS UI → Datanodes tab: confirm block counts are not 100% identical across every
  node once you're at 3+ nodes with replication < node count.

## Idempotency check before trusting a new node

Before running `setup_node.sh` against a machine you've never touched before,
re-run it against an existing, already-configured node first and confirm nothing
breaks or gets duplicated (e.g. `SPARK_LOCAL_IP` appended twice to `.bashrc`,
NameNode re-formatted and wiping data). This is the single highest-leverage check
in the whole setup — validate it early, while stakes are low.
