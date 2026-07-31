#!/usr/bin/env bash
# cluster/setup_node.sh
#
# Brings a machine up as a Spark worker + HDFS DataNode, or (with role=master)
# as the Spark master + HDFS NameNode. Designed to be idempotent — safe to
# re-run against a machine that's already set up (validate this against your
# existing nodes before trusting it for a brand new one).
#
# Usage:
#   ./setup_node.sh master   # run once, on the coordinator machine
#   ./setup_node.sh worker   # run on every other machine, including new ones
#
# Assumes:
#   - Tailscale + MagicDNS already configured, this machine already resolves
#     by its hostname (see cluster/nodes.txt for the canonical list).
#   - HADOOP_HOME and SPARK_HOME already point at matching installs across
#     all machines (same versions everywhere — check before running).
#   - Passwordless SSH is set up from the master to every worker (needed for
#     start-dfs.sh / start-master.sh to reach other machines).

set -euo pipefail

ROLE="${1:-}"
NODES_FILE="$(dirname "$0")/nodes.txt"
MASTER_HOST="${MASTER_HOST:-laptop-a}"   # override via env if your master name differs

if [[ "$ROLE" != "master" && "$ROLE" != "worker" ]]; then
  echo "Usage: $0 <master|worker>"
  exit 1
fi

if [[ ! -f "$NODES_FILE" ]]; then
  echo "ERROR: $NODES_FILE not found. This is the canonical node list — create it first."
  exit 1
fi

echo "== Step 1: Set SPARK_LOCAL_IP to this machine's own resolvable name =="
THIS_HOST="$(hostname)"
if ! grep -qxF "export SPARK_LOCAL_IP=" ~/.bashrc 2>/dev/null; then
  echo "export SPARK_LOCAL_IP=${THIS_HOST}" >> ~/.bashrc
  echo "  Added SPARK_LOCAL_IP=${THIS_HOST} to ~/.bashrc"
else
  echo "  SPARK_LOCAL_IP already set in ~/.bashrc, skipping (idempotent)"
fi
export SPARK_LOCAL_IP="${THIS_HOST}"

echo "== Step 2: Sync Hadoop 'workers' file from canonical cluster/nodes.txt =="
cp "$NODES_FILE" "${HADOOP_HOME}/etc/hadoop/workers"
echo "  Synced $(wc -l < "$NODES_FILE") node(s) into \$HADOOP_HOME/etc/hadoop/workers"

echo "== Step 3: Verify core-site.xml / hdfs-site.xml point at the master =="
CORE_SITE="${HADOOP_HOME}/etc/hadoop/core-site.xml"
if ! grep -q "${MASTER_HOST}:9000" "$CORE_SITE" 2>/dev/null; then
  echo "  WARNING: ${CORE_SITE} does not reference hdfs://${MASTER_HOST}:9000"
  echo "  Fix this manually before continuing — see cluster/README.md."
fi

if [[ "$ROLE" == "master" ]]; then
  echo "== Step 4 (master): format namenode if not already formatted =="
  NN_DIR="${HADOOP_HOME}/data/nameNode"
  if [[ ! -d "$NN_DIR" || -z "$(ls -A "$NN_DIR" 2>/dev/null)" ]]; then
    hdfs namenode -format -nonInteractive
  else
    echo "  NameNode dir already formatted, skipping (idempotent)"
  fi

  echo "== Step 5 (master): start HDFS + Spark master =="
  start-dfs.sh
  "${SPARK_HOME}/sbin/start-master.sh"

  echo "== Done. Check: =="
  echo "    HDFS UI:  http://${THIS_HOST}:9870"
  echo "    Spark UI: http://${THIS_HOST}:8080"
else
  echo "== Step 4 (worker): register with master =="
  echo "  NOTE: DataNode + Spark worker daemons are normally started FROM the"
  echo "  master via start-dfs.sh (which SSHes out to every host in 'workers')"
  echo "  and by running start-worker.sh manually on this machine pointed at"
  echo "  the master. Run on THIS machine:"
  echo ""
  echo "    ${SPARK_HOME}/sbin/start-worker.sh spark://${MASTER_HOST}:7077"
  echo ""
  echo "  Then, from the MASTER machine, re-run start-dfs.sh (or, if this node"
  echo "  was added after the cluster was already running, run:"
  echo "    hdfs dfsadmin -refreshNodes"
  echo "  from the master) so the new DataNode is picked up."
fi

echo ""
echo "== Reminder: after adding ANY new node to an already-running cluster =="
echo "   run 'hdfs balancer' from the master to rebalance existing blocks"
echo "   onto the new node — this does NOT happen automatically."
