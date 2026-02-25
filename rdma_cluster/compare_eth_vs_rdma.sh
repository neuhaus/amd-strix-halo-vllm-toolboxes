#!/usr/bin/env bash

# -------- dynamic config --------
HOST_ROCE="192.168.100.2"
HOST_ETH="192.168.1.127"
HOST_TB="192.168.2.2"

# Parse args
RUN_ETH=true
RUN_ROCE=true
RUN_TB=true
RUN_RDMA=true

# If any flags are provided, turn off defaults and only run requested
if [ "$#" -gt 0 ]; then
    RUN_ETH=false
    RUN_ROCE=false
    RUN_TB=false
    RUN_RDMA=false
fi

while getopts "ertih" opt; do
    case ${opt} in
        e ) RUN_ETH=true ;;
        r ) RUN_ROCE=true ;;
        t ) RUN_TB=true ;;
        i ) RUN_RDMA=true ;;
        h ) echo "Usage: $0 [-e (Ethernet LAN)] [-r (RoCE Ethernet/TCP)] [-t (Thunderbolt)] [-i (RDMA/Infiniband)]"
            echo
            echo "Options:"
            echo "  -e    Run benchmarking for standard Ethernet (1G LAN)."
            echo "  -r    Run benchmarking for RoCE NIC (via Ethernet/TCP)."
            echo "  -t    Run benchmarking for Thunderbolt link."
            echo "  -i    Run benchmarking for RDMA (RoCE v2)."
            echo "  -h    Print this help message and exit."
            echo
            echo "If no arguments are provided, all benchmarks are executed."
            exit 0
            ;;
        \? ) echo "Usage: cmd [-e (Ethernet LAN)] [-r (RoCE Ethernet/TCP)] [-t (Thunderbolt)] [-i (RDMA/Infiniband)] [-h (Help)]"
             exit 1
             ;;
    esac
done

# Automatically detect local and remote RDMA device names if needed
if [ "$RUN_RDMA" = true ]; then
    RDMA_DEV_LOCAL=$(ibv_devices | awk 'NR==3 {print $1}')
    RDMA_DEV_REMOTE=$(ssh "$HOST_ROCE" "toolbox run -c vllm -- ibv_devices | awk 'NR==3 {print \$1}'")
fi

WORKDIR="/tmp/rdma_bench"
mkdir -p "$WORKDIR"

# -------- helpers --------
parse_ping_avg() {
    if [ -f "$1" ]; then
        grep rtt "$1" | awk -F'/' '{print $5}'
    else
        echo "0"
    fi
}

parse_iperf_gbps() {
    if [ -f "$1" ]; then
        grep receiver "$1" | tail -n1 | awk '
        {
            val=$(NF-2);
            unit=$(NF-1);
            if (unit=="Mbits/sec") printf "%.2f", val/1000;
            else if (unit=="Gbits/sec") printf "%.2f", val;
            else print "0.00";
        }'
    else
        echo "0.00"
    fi
}

parse_rdma_lat_us() {
    if [ -f "$1" ]; then
        val=$(grep -E '^[[:space:]]*[0-9]+' "$1" | tail -n1 | awk '{print $6}')
        echo "${val:-0}"
    else
        echo "0"
    fi
}

parse_rdma_bw_mib() {
    if [ -f "$1" ]; then
        val=$(grep -E '^[[:space:]]*[0-9]+' "$1" | tail -n1 | awk '{print $4}')
        echo "${val:-0}"
    else
        echo "0"
    fi
}

# Clear old results
rm -f "$WORKDIR"/*.txt

if [ "$RUN_ETH" = true ]; then
    # -------- normal ethernet --------
    echo "[*] Benchmarking Ethernet (1G LAN)..."
    ping -c 10 "$HOST_ETH" > "$WORKDIR/ping_eth.txt"
    ssh "$HOST_ROCE" "toolbox run -c vllm -- iperf3 -s -1" >/dev/null 2>&1 &
    sleep 1
    iperf3 -c "$HOST_ETH" -P 8 -t 10 > "$WORKDIR/iperf_eth.txt"
fi

if [ "$RUN_ROCE" = true ]; then
    # -------- roce ethernet (tcp) --------
    echo "[*] Benchmarking RoCE NIC (Ethernet/TCP)..."
    ping -c 10 "$HOST_ROCE" > "$WORKDIR/ping_roce.txt"
    ssh "$HOST_ROCE" "toolbox run -c vllm -- iperf3 -s -1" >/dev/null 2>&1 &
    sleep 1
    iperf3 -c "$HOST_ROCE" -P 8 -t 10 > "$WORKDIR/iperf_roce.txt"
fi

if [ "$RUN_TB" = true ]; then
    # -------- thunderbolt ethernet (tcp) --------
    echo "[*] Benchmarking Thunderbolt..."
    ping -c 10 "$HOST_TB" > "$WORKDIR/ping_tb.txt"
    ssh "$HOST_TB" "toolbox run -c vllm -- iperf3 -s -1" >/dev/null 2>&1 &
    sleep 1
    iperf3 -c "$HOST_TB" -P 8 -t 10 > "$WORKDIR/iperf_tb.txt"
fi

if [ "$RUN_RDMA" = true ]; then
    # -------- rdma latency --------
    echo "[*] Benchmarking RDMA (RoCE v2)..."
    ssh "$HOST_ROCE" "toolbox run -c vllm -- ib_send_lat --rdma_cm -d $RDMA_DEV_REMOTE" > "$WORKDIR/rdma_lat_srv.txt" 2>&1 &
    sleep 2
    ib_send_lat --rdma_cm -d "$RDMA_DEV_LOCAL" "$HOST_ROCE" > "$WORKDIR/rdma_lat_cli.txt" 2>&1

    # -------- rdma bandwidth (maximized) --------
    # We use -x 1 because show_gids confirmed RoCE v2 is at Index 1
    ssh "$HOST_ROCE" "toolbox run -c vllm -- ib_write_bw -a -x 1 -q 8 -m 4096" > "$WORKDIR/rdma_bw_srv.txt" 2>&1 &
    sleep 2
    ib_write_bw -a -x 1 -q 8 -m 4096 "$HOST_ROCE" > "$WORKDIR/rdma_bw_cli.txt" 2>&1
fi

# -------- parse --------
ETH_LAT_MS=$(parse_ping_avg "$WORKDIR/ping_eth.txt")
ETH_BW=$(parse_iperf_gbps "$WORKDIR/iperf_eth.txt")

ROCE_LAT_MS=$(parse_ping_avg "$WORKDIR/ping_roce.txt")
ROCE_BW=$(parse_iperf_gbps "$WORKDIR/iperf_roce.txt")

TB_LAT_MS=$(parse_ping_avg "$WORKDIR/ping_tb.txt")
TB_BW=$(parse_iperf_gbps "$WORKDIR/iperf_tb.txt")

RDMA_LAT_US=$(parse_rdma_lat_us "$WORKDIR/rdma_lat_cli.txt")
RDMA_BW_MIB=$(parse_rdma_bw_mib "$WORKDIR/rdma_bw_cli.txt")

# Convert units for dual display
ETH_LAT_US=$(python3 -c "print(f'{float(${ETH_LAT_MS:-0}) * 1000:.2f}')" 2>/dev/null || echo "0.00")
ROCE_LAT_US=$(python3 -c "print(f'{float(${ROCE_LAT_MS:-0}) * 1000:.2f}')" 2>/dev/null || echo "0.00")
TB_LAT_US=$(python3 -c "print(f'{float(${TB_LAT_MS:-0}) * 1000:.2f}')" 2>/dev/null || echo "0.00")
RDMA_LAT_MS=$(python3 -c "print(f'{float(${RDMA_LAT_US:-0}) / 1000:.3f}')" 2>/dev/null || echo "0.00")

RDMA_BW_GBPS=$(python3 - <<EOF
import sys
try:
    print(round($RDMA_BW_MIB * 8 / 1024, 2))
except:
    print("0.00")
EOF
)

# -------- output --------
echo
echo "=== Network Comparison ==="
echo
printf "%-25s %-15s %-15s %-12s\n" "Path" "Latency (ms)" "Latency (us)" "Bandwidth"
echo "-----------------------------------------------------------------------"
if [ "$RUN_ETH" = true ]; then
    printf "%-25s %-15s %-15s %-12s\n" "Ethernet (1G LAN)" "${ETH_LAT_MS:-0.00} ms" "${ETH_LAT_US:-0.00} us" "${ETH_BW:-0.00} Gbps"
fi
if [ "$RUN_ROCE" = true ]; then
    printf "%-25s %-15s %-15s %-12s\n" "Ethernet (RoCE NIC)" "${ROCE_LAT_MS:-0.00} ms" "${ROCE_LAT_US:-0.00} us" "${ROCE_BW:-0.00} Gbps"
fi
if [ "$RUN_TB" = true ]; then
    printf "%-25s %-15s %-15s %-12s\n" "Ethernet (Thunderbolt)" "${TB_LAT_MS:-0.00} ms" "${TB_LAT_US:-0.00} us" "${TB_BW:-0.00} Gbps"
fi
if [ "$RUN_RDMA" = true ]; then
    printf "%-25s %-15s %-15s %-12s\n" "RDMA (RoCE)" "${RDMA_LAT_MS:-0.00} ms" "${RDMA_LAT_US:-0.00} us" "${RDMA_BW_GBPS:-0.00} Gbps"
fi
echo
