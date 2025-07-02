import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict, deque

# Page configuration (needs to be at the very top)
st.set_page_config(page_title="OSync Simulator", layout="centered")

# Custom CSS for general styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=Cursive');

    .classic-title {
        font-family: "Lucida Console", "Courier New", monospace;
        font-size: 40px;
        color: #ff2c2c;
        text-align: center;
        font-weight: 500;
        letter-spacing: 1.0px;
        margin-bottom: 25px;
        padding-bottom: 12px;
        background: linear-gradient(90deg, #004e92, #000428);
        text-shadow:
            1px 1px 2px rgba(0, 0, 0, 0.2),
            2px 2px 4px rgba(0, 0, 0, 0.15);
        border-bottom: 2px dashed #777;
        transition: all 0.3s ease-in-out;
    }
    .classic-title:hover {
        transform: scale(1.03);
        letter-spacing: 1.5px;
    }

    /* Main button styling */
    .stButton>button {
        font-family: "Lucida Console", "Courier New", monospace;
        background-color: #007BFF;
        color: white;
        font-size: 18px;
        font-weight: 500;
        border-radius: 12px;
        border: none;
        padding: 12px 30px;
        width: 100%;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        font-size: 20px; /* Slightly larger on hover */
        transform: scale(1.03);
    }

    /* Sidebar button styling for better appearance */
    .st-emotion-cache-vk33gh button { /* Targeting buttons inside sidebar more specifically */
        width: 100%;
        margin-bottom: 10px; /* Space out sidebar buttons */
        background-color: #004e92; /* Darker blue for sidebar buttons */
    }
    .st-emotion-cache-vk33gh button:hover {
        background-color: #003a70; /* Even darker on hover */
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Title and Image (Moved to the very top of the body) ---
st.markdown("""
    <h1 class="classic-title">🔀 OSync: Page Replacement & Scheduling Simulator</h1>
""", unsafe_allow_html=True)
# Ensure you have an 'img' folder in the same directory as your script and 'back.jpg' inside it.
st.image('img/back.jpg', use_container_width=True)
st.markdown("<h4 style='text-align:center; color: gray;'>Visualize. Simulate. Learn.</h4>", unsafe_allow_html=True)

# Initialize session state for page and theory display mode
if "page" not in st.session_state:
    st.session_state.page = "home"
if "theory_display_mode" not in st.session_state:
    st.session_state.theory_display_mode = "all" # Default to showing all theory

def go_to_page(name):
    st.session_state.page = name
    # Update theory_display_mode based on the destination page
    if name == "home":
        st.session_state.theory_display_mode = "all"
    elif name == "page_replacement":
        st.session_state.theory_display_mode = "page_replacement"
    elif name == "process_scheduling":
        st.session_state.theory_display_mode = "process_scheduling"
    # If navigating to 'theory' page, the theory_display_mode will be based on the last simulator visited
    # or "all" if starting from home.

# --- Navigation Sidebar ---
with st.sidebar:
    st.header("Navigation")
    if st.button("🏠 Home", key="nav_home_sidebar"):
        go_to_page("home")
    if st.button("📄 Page Replacement Simulator", key="nav_page_replacement_sidebar"):
        go_to_page("page_replacement")
    if st.button("📊 Process Scheduling Simulator", key="nav_process_scheduling_sidebar"):
        go_to_page("process_scheduling")
    st.markdown("---")
    if st.button("📚 Concepts & Theory", key="nav_theory_sidebar"):
        go_to_page("theory")

# --- Page Content ---

# HOME PAGE
if st.session_state.page == "home":
    st.subheader("Your Interactive OS Learning Tool")
    st.markdown("""
    **OSync** is a simulator designed to help students understand **Page Replacement Algorithms** and **CPU Scheduling Algorithms**.

    It supports:
    - **Page Replacement:** FIFO, LRU, LFU
    - **CPU Scheduling:** Round Robin, Shortest Job First (SJF), First-Come, First-Served (FCFS)

    Use the navigation sidebar on the left to explore different simulators and learn about the underlying theories.
    """)
    


# PAGE REPLACEMENT SIMULATOR
elif st.session_state.page == "page_replacement":
    st.header("📄 Page Replacement Simulator")

    algo = st.selectbox("Choose Algorithm", ["FIFO", "LRU", "LFU"])
    num_pages = st.number_input("Enter number of pages (This input is not directly used for logic but kept for consistency)", min_value=1, value=8)
    pages = st.text_input("Page reference string (comma-separated)", placeholder="7, 0, 1, 2, 0, 3, 0, 4")
    frames = st.number_input("Enter number of frames", min_value=1, value=3)

    # Recommendation Block for Page Replacement
    if pages.strip() and frames > 0:
        try:
            page_list = list(map(int, pages.split(',')))
            frame_count = int(frames)

            def get_faults(algorithm, ref, frames):
                memory = []
                faults = 0
                recent = {}
                queue = deque()
                frequency = defaultdict(int)
                page_arrival_time_in_frame = {} 

                for i, page in enumerate(ref):
                    if page in memory:
                        if algorithm == "LRU":
                            recent[page] = i
                        if algorithm == "LFU":
                            frequency[page] += 1
                        continue

                    faults += 1
                    if len(memory) < frames:
                        memory.append(page)
                        if algorithm == "FIFO":
                            queue.append(page)
                        if algorithm == "LFU":
                            page_arrival_time_in_frame[page] = i 
                    else:
                        if algorithm == "FIFO":
                            oldest_page = queue.popleft()
                            memory.remove(oldest_page) 
                        elif algorithm == "LRU":
                            lru_page = min(recent, key=recent.get)
                            memory.remove(lru_page)
                            del recent[lru_page]
                        elif algorithm == "LFU":
                            min_freq = float('inf')
                            lfu_candidates = []
                            for p in memory:
                                if frequency[p] < min_freq:
                                    min_freq = frequency[p]
                                    lfu_candidates = [p]
                                elif frequency[p] == min_freq:
                                    lfu_candidates.append(p)

                            page_to_remove = min(lfu_candidates, key=lambda p: page_arrival_time_in_frame[p])

                            memory.remove(page_to_remove)
                            del frequency[page_to_remove]
                            del page_arrival_time_in_frame[page_to_remove]

                        memory.append(page)
                        if algorithm == "FIFO":
                            queue.append(page)
                        if algorithm == "LFU":
                            page_arrival_time_in_frame[page] = i

                    if algorithm == "LRU":
                        recent[page] = i
                    if algorithm == "LFU":
                        frequency[page] += 1
                return faults

            fifo_faults = get_faults("FIFO", page_list, frame_count)
            lru_faults = get_faults("LRU", page_list, frame_count)
            lfu_faults = get_faults("LFU", page_list, frame_count)

            fault_counts = {"FIFO": fifo_faults, "LRU": lru_faults, "LFU": lfu_faults}
            best_algo = min(fault_counts, key=fault_counts.get)
            min_faults = fault_counts[best_algo]

            tied_algos = [algo_name for algo_name, faults_num in fault_counts.items() if faults_num == min_faults]

            if len(tied_algos) > 1:
                recommendation = (f"Based on your input, **{', '.join(tied_algos)}** performed equally well "
                                  f"with {min_faults} page faults.")
                if "LRU" in tied_algos or "LFU" in tied_algos:
                    recommendation += " LRU and LFU are generally more adaptive for realistic workloads."
            elif min_faults == 0:
                recommendation = "Page fault count is 0 for all algorithms. No replacement needed!"
            else:
                recommendation = f"Based on your input, **{best_algo}** is likely to perform best."

            st.markdown(f"""
            ### 🤖 Recommendation:
            {recommendation}
            - FIFO Page Faults: `{fifo_faults}`
            - LRU Page Faults: `{lru_faults}`
            - LFU Page Faults: `{lfu_faults}`
            """)

        except Exception as e:
            st.warning(f"Could not analyze best algorithm. Please check your input. Error: {e}")

    if st.button("Simulate", key="page_simulate_btn"):
        if not pages.strip():
            st.error("Please enter a valid page reference string.")
        else:
            try:
                page_list = list(map(int, pages.split(',')))
                frame_count = int(frames)

                timeline = []
                faults = 0
                hits = 0
                frame_history = []
                display_labels = []

                if algo == "FIFO":
                    frame = deque(maxlen=frame_count)
                    for i, page in enumerate(page_list):
                        is_hit = False
                        if page in frame:
                            hits += 1
                            is_hit = True
                        else:
                            faults += 1
                            if len(frame) == frame_count:
                                frame.popleft()
                            frame.append(page)
                        frame_history.append(list(frame))
                        display_labels.append(f"{'Hit' if is_hit else 'Fault'}: {page}")

                elif algo == "LRU":
                    frame = []
                    recent = {}
                    for i, page in enumerate(page_list):
                        is_hit = False
                        if page in frame:
                            hits += 1
                            is_hit = True
                        else:
                            faults += 1
                            if len(frame) < frame_count:
                                frame.append(page)
                            else:
                                lru_page = min(recent, key=recent.get)
                                frame[frame.index(lru_page)] = page
                                del recent[lru_page]
                        recent[page] = i
                        frame_history.append(frame.copy())
                        display_labels.append(f"{'Hit' if is_hit else 'Fault'}: {page}")

                elif algo == "LFU":
                    frame = []
                    frequency = defaultdict(int)
                    page_arrival_time_in_frame = {}

                    for i, page in enumerate(page_list):
                        is_hit = False
                        if page in frame:
                            hits += 1
                            is_hit = True
                        else:
                            faults += 1
                            if len(frame) < frame_count:
                                frame.append(page)
                                page_arrival_time_in_frame[page] = i
                            else:
                                min_freq = float('inf')
                                lfu_candidates = []
                                for p in frame:
                                    if frequency[p] < min_freq:
                                        min_freq = frequency[p]
                                        lfu_candidates = [p]
                                    elif frequency[p] == min_freq:
                                        lfu_candidates.append(p)

                                page_to_remove = min(lfu_candidates, key=lambda p: page_arrival_time_in_frame[p])

                                memory.remove(page_to_remove)
                                del frequency[page_to_remove]
                                del page_arrival_time_in_frame[page_to_remove]

                                page_arrival_time_in_frame[page] = i
                        
                        frequency[page] += 1
                        frame_history.append(frame.copy())
                        display_labels.append(f"{'Hit' if is_hit else 'Fault'}: {page}")

                st.success(f"Simulation complete! Total Page Faults: {faults}, Hits: {hits}")

                st.markdown("### 🔍 Frame History Table")
                df_data = []
                for step, (frame_state, label) in enumerate(zip(frame_history, display_labels)):
                    row = [f"Step {step+1} ({label})"] + frame_state + ['-'] * (frame_count - len(frame_state))
                    df_data.append(row)

                df = pd.DataFrame(df_data, columns=["Step/Event"] + [f"Frame {i+1}" for i in range(frame_count)])
                st.dataframe(df)

                st.markdown("### 📈 Matplotlib Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                for i in range(frame_count):
                    y_values = [f[i] if i < len(f) else None for f in frame_history]
                    x_values = [j for j, y in enumerate(y_values) if y is not None]
                    y_values_filtered = [y for y in y_values if y is not None]
                    ax.plot(x_values, y_values_filtered, label=f'Frame {i+1}', marker='o')

                ax.set_title(f"{algo} Page Replacement")
                ax.set_xlabel("Steps")
                ax.set_ylabel("Page Number")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                st.markdown("### 📊 Plotly Timeline")
                fig2 = go.Figure()
                for i in range(frame_count):
                    y_values = [f[i] if i < len(f) else None for f in frame_history]
                    x_values = [f"Step {j+1}" for j, y in enumerate(y_values) if y is not None]
                    y_values_filtered = [y for y in y_values if y is not None]
                    fig2.add_trace(go.Scatter(
                        x=x_values,
                        y=y_values_filtered,
                        mode='lines+markers',
                        name=f"Frame {i+1}"
                    ))
                fig2.update_layout(title=f"{algo} Frame Timeline", xaxis_title="Step", yaxis_title="Page")
                st.plotly_chart(fig2)

            except Exception as e:
                st.error(f"Error in simulation: {e}")



# PROCESS SCHEDULING SIMULATOR
elif st.session_state.page == "process_scheduling":
    st.header("📊 Process Scheduling Simulator")

    sched_algo = st.selectbox("Choose Algorithm", ["Round Robin", "Shortest Job First (SJF)", "First-Come, First-Served (FCFS)"])
    num_processes = st.number_input("Enter number of processes", min_value=1, value=3)
    burst_input = st.text_area("Enter burst times (comma-separated)", placeholder="5, 3, 8")

    if sched_algo == "Round Robin":
        quantum = st.number_input("Enter time quantum", min_value=1, value=2)

    # Recommendation Block for Process Scheduling
    if burst_input.strip() and num_processes > 0:
        try:
            burst_times_str = burst_input.split(',')
            burst_times = [int(bt.strip()) for bt in burst_times_str if bt.strip()]

            if len(burst_times) == num_processes:
                def fcfs_process_scheduling_calc(burst_times):
                    waiting_time = [0] * len(burst_times)
                    turnaround_time = [0] * len(burst_times)
                    current_time = 0
                    for i, bt in enumerate(burst_times):
                        waiting_time[i] = current_time
                        turnaround_time[i] = current_time + bt
                        current_time += bt
                    return sum(waiting_time) / len(waiting_time), sum(turnaround_time) / len(turnaround_time)

                def round_robin_process_scheduling_calc(burst_times, quantum):
                    n = len(burst_times)
                    remaining = list(burst_times)
                    waiting_time = [0] * n
                    completion_time = [0] * n
                    t = 0
                    queue = deque(range(n))
                    last_execution_start_time = [0] * n

                    while any(remaining) or queue:
                        current_queue_size = len(queue)
                        if current_queue_size == 0 and any(r > 0 for r in remaining):
                            for i in range(n):
                                if remaining[i] > 0:
                                    queue.append(i)
                                    break
                            if not queue:
                                break
                            
                        if not queue:
                            t += 1
                            continue

                        i = queue.popleft()
                        
                        if remaining[i] > 0:
                            waiting_time[i] += t - last_execution_start_time[i]
                            
                            exec_time = min(quantum, remaining[i])
                            
                            start_segment_time = t
                            t += exec_time
                            remaining[i] -= exec_time
                            last_execution_start_time[i] = t

                            if remaining[i] > 0:
                                queue.append(i)
                            else:
                                completion_time[i] = t
                    
                    final_turnaround = [completion_time[i] for i in range(n)]
                    final_waiting = [turnaround - burst_times[i] for i, turnaround in enumerate(final_turnaround)]

                    avg_wait = sum(final_waiting) / n if n > 0 else 0
                    avg_turn = sum(final_turnaround) / n if n > 0 else 0
                    return avg_wait, avg_turn


                def sjf_process_scheduling_calc(burst_times):
                    processes_with_ids = [(i, bt) for i, bt in enumerate(burst_times)]
                    sorted_processes = sorted(processes_with_ids, key=lambda x: x[1])
                    
                    time = 0
                    waiting_time = [0] * len(burst_times)
                    turnaround_time = [0] * len(burst_times)
                    
                    for original_idx, bt in sorted_processes:
                        waiting_time[original_idx] = time
                        turnaround_time[original_idx] = time + bt
                        time += bt
                    return sum(waiting_time) / len(waiting_time), sum(turnaround_time) / len(turnaround_time)

                # Calculate metrics for all algorithms
                metrics = {}
                if num_processes > 0:
                    try:
                        avg_wait_fcfs, avg_turn_fcfs = fcfs_process_scheduling_calc(burst_times)
                        metrics["FCFS"] = {"Wait": avg_wait_fcfs, "Turn": avg_turn_fcfs}
                    except Exception as e:
                        st.error(f"Error calculating FCFS metrics: {e}")
                    
                    try:
                        if 'quantum' in locals() and quantum > 0:
                            avg_wait_rr, avg_turn_rr = round_robin_process_scheduling_calc(burst_times, quantum)
                            metrics["Round Robin"] = {"Wait": avg_wait_rr, "Turn": avg_turn_rr}
                        elif sched_algo == "Round Robin":
                             st.warning("Please enter a positive time quantum for Round Robin calculation for recommendation.")
                    except Exception as e:
                        st.error(f"Error calculating Round Robin metrics: {e}")
                    
                    try:
                        avg_wait_sjf, avg_turn_sjf = sjf_process_scheduling_calc(burst_times)
                        metrics["SJF"] = {"Wait": avg_wait_sjf, "Turn": avg_turn_sjf}
                    except Exception as e:
                        st.error(f"Error calculating SJF metrics: {e}")

                if metrics:
                    st.markdown("### 🤖 Recommendation for CPU Scheduling:")
                    
                    best_wait_algo = None
                    min_avg_wait = float('inf')
                    for algo_name, vals in metrics.items():
                        if vals["Wait"] < min_avg_wait:
                            min_avg_wait = vals["Wait"]
                            best_wait_algo = algo_name
                        elif vals["Wait"] == min_avg_wait:
                            if "SJF" in [best_wait_algo, algo_name]:
                                best_wait_algo = "SJF"
                            elif best_wait_algo in ["FCFS", "Round Robin"] and algo_name in ["FCFS", "Round Robin"]:
                                if "Round Robin" in [best_wait_algo, algo_name]:
                                    best_wait_algo = "Round Robin"


                    best_turn_algo = None
                    min_avg_turn = float('inf')
                    for algo_name, vals in metrics.items():
                        if vals["Turn"] < min_avg_turn:
                            min_avg_turn = vals["Turn"]
                            best_turn_algo = algo_name
                        elif vals["Turn"] == min_avg_turn:
                            if "SJF" in [best_turn_algo, algo_name]:
                                best_turn_algo = "SJF"
                            elif best_turn_algo in ["FCFS", "Round Robin"] and algo_name in ["FCFS", "Round Robin"]:
                                if "Round Robin" in [best_turn_algo, algo_name]:
                                    best_turn_algo = "Round Robin"

                    if best_wait_algo:
                        st.markdown(f"Based on **Average Waiting Time**, **{best_wait_algo}** performs best (Avg. Wait: `{metrics[best_wait_algo]['Wait']:.2f}`).")
                    if best_turn_algo:
                        st.markdown(f"Based on **Average Turnaround Time**, **{best_turn_algo}** performs best (Avg. Turn: `{metrics[best_turn_algo]['Turn']:.2f}`).")
                    
                    st.markdown("---")
                    st.markdown("#### Performance Metrics for all Algorithms:")
                    for algo_name, vals in metrics.items():
                        st.markdown(f"- **{algo_name}:** Average Waiting Time = `{vals['Wait']:.2f}`, Average Turnaround Time = `{vals['Turn']:.2f}`")

                    st.info("💡 **Note:** SJF often yields the best average waiting/turnaround times, but it requires knowing future burst times. Round Robin provides fairness in time-sharing systems.")
                else:
                    st.warning("Could not calculate metrics for all algorithms. Please check your inputs.")

        except Exception as e:
            st.warning(f"Could not analyze best algorithm. Please check your input. Error: {e}")

    if st.button("Simulate", key="process_simulate_btn"):
        if not burst_input.strip():
            st.error("Please enter valid burst times.")
        else:
            try:
                burst_times_str = burst_input.split(',')
                burst_times = [int(bt.strip()) for bt in burst_times_str if bt.strip()]

                if len(burst_times) != num_processes:
                    st.error(f"Please enter exactly {num_processes} burst times.")
                else:
                    st.success(f"Running {sched_algo}...")

                    def fcfs_process_scheduling(burst_times):
                        timeline = []
                        waiting_time = [0] * len(burst_times)
                        turnaround_time = [0] * len(burst_times)
                        current_time = 0

                        for i, bt in enumerate(burst_times):
                            start_time = current_time
                            end_time = current_time + bt
                            timeline.append((f"P{i+1}", start_time, end_time))
                            waiting_time[i] = current_time
                            turnaround_time[i] = end_time
                            current_time = end_time
                        return timeline, waiting_time, turnaround_time

                    def round_robin_process_scheduling(burst_times, quantum):
                        n = len(burst_times)
                        remaining = list(burst_times)
                        waiting_time = [0] * n
                        completion_time = [0] * n
                        t = 0
                        timeline = []
                        queue = deque(range(n))
                        last_execution_start_time = [0] * n

                        while any(remaining) or queue:
                            current_queue_size = len(queue)
                            if current_queue_size == 0 and any(r > 0 for r in remaining):
                                for i in range(n):
                                    if remaining[i] > 0:
                                        queue.append(i)
                                        break
                                if not queue:
                                    break
                                
                            if not queue:
                                t += 1
                                continue

                            i = queue.popleft()
                            
                            if remaining[i] > 0:
                                waiting_time[i] += t - last_execution_start_time[i]
                                
                                exec_time = min(quantum, remaining[i])
                                
                                start_segment_time = t
                                t += exec_time
                                remaining[i] -= exec_time
                                last_execution_start_time[i] = t

                                timeline.append((f"P{i+1}", start_segment_time, t))

                                if remaining[i] > 0:
                                    queue.append(i)
                                else:
                                    completion_time[i] = t
                        
                        final_turnaround = [completion_time[i] for i in range(n)]
                        final_waiting = [final_turnaround[i] - burst_times[i] for i, turnaround in enumerate(final_turnaround)]

                        return timeline, final_waiting, final_turnaround


                    def sjf_process_scheduling(burst_times):
                        processes_with_ids = [(i, bt) for i, bt in enumerate(burst_times)]
                        sorted_processes = sorted(processes_with_ids, key=lambda x: x[1])
                        
                        time = 0
                        timeline = []
                        waiting_time = [0] * len(burst_times)
                        turnaround = [0] * len(burst_times)
                        
                        for original_idx, bt in sorted_processes:
                            timeline.append((f"P{original_idx+1}", time, time + bt))
                            waiting_time[original_idx] = time
                            turnaround[original_idx] = time + bt
                            time += bt
                        return timeline, waiting_time, turnaround

                    if sched_algo == "Round Robin":
                        timeline, waiting, turnaround = round_robin_process_scheduling(burst_times, quantum)
                    elif sched_algo == "Shortest Job First (SJF)":
                        timeline, waiting, turnaround = sjf_process_scheduling(burst_times)
                    elif sched_algo == "First-Come, First-Served (FCFS)":
                        timeline, waiting, turnaround = fcfs_process_scheduling(burst_times)

                    st.markdown("### 📋 Process Timeline Table")
                    df = pd.DataFrame(timeline, columns=["Process", "Start Time", "End Time"])
                    st.dataframe(df)

                    st.markdown("### 📈 Matplotlib Visualization")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    process_labels = sorted(list(set([t[0] for t in timeline])))
                    
                    y_coords = {pid: idx for idx, pid in enumerate(process_labels)}

                    for pid, start, end in timeline:
                        ax.barh(y_coords[pid], end - start, left=start, height=0.6, label=pid if pid not in ax.artists else "")
                        ax.text((start + end)/2, y_coords[pid], pid, ha='center', va='center', color='white', fontweight='bold')
                    
                    ax.set_yticks(list(y_coords.values()))
                    ax.set_yticklabels(list(y_coords.keys()))
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Processes")
                    ax.set_title(f"{sched_algo} Scheduling Gantt Chart")
                    ax.grid(axis='x', linestyle='--')
                    st.pyplot(fig)


                    st.markdown("### 📊 Plotly Timeline")
                    fig2 = go.Figure()
                    
                    processes_in_order = sorted(list(set([item[0] for item in timeline])))

                    for pid, start, end in timeline:
                        fig2.add_trace(go.Bar(
                            x=[end - start],
                            y=[pid],
                            base=[start],
                            orientation='h',
                            name=pid,
                            hoverinfo='text',
                            hovertext=f"Process: {pid}<br>Start: {start}<br>End: {end}<br>Duration: {end-start}"
                        ))
                    fig2.update_layout(
                        barmode='stack',
                        title=f"{sched_algo} Execution Timeline",
                        xaxis_title="Time",
                        yaxis_title="Process",
                        yaxis=dict(categoryorder='array', categoryarray=processes_in_order),
                        showlegend=False
                    )
                    st.plotly_chart(fig2)

                    if sum(burst_times) > 0 and len(waiting) > 0 and len(turnaround) > 0:
                        avg_wait = sum(waiting) / len(waiting)
                        avg_turn = sum(turnaround) / len(turnaround)
                    else:
                        avg_wait = 0
                        avg_turn = 0
                    
                    st.markdown(f"""
                    ### Performance for selected algorithm ({sched_algo}):
                    - Average Waiting Time: `{avg_wait:.2f}`
                    - Average Turnaround Time: `{avg_turn:.2f}`
                    """)

            except ValueError:
                st.error("Invalid burst times input. Please ensure the input is comma-separated integers.")
            except Exception as e:
                st.error(f"An unexpected error occurred during simulation: {e}")



# ALGORITHMS THEORY PAGE
elif st.session_state.page == "theory":
    st.header("📚 Concepts & Theory")

    st.markdown("""
    This section provides a brief explanation of the Page Replacement and CPU Scheduling algorithms used in this simulator.
    The content displayed below adapts based on which simulator tab you last visited.
    """)

    # Conditionally display theory content
    if st.session_state.theory_display_mode == "page_replacement":
        st.info("Displaying theory for **Page Replacement Algorithms**.")
        st.subheader("📄 Page Replacement Algorithms")
        
        with st.expander("FIFO (First-In, First-Out) Page Replacement"):
            st.markdown("""
            **Concept:** The simplest page replacement algorithm. It replaces the oldest page in memory, regardless of how recently or how frequently it has been used. It's like a queue where the page that has been in memory the longest is the first one out.

            **How it Works:**
            * When a page fault occurs and memory is full, the page that was loaded earliest into memory is removed.
            * A queue data structure is typically used to keep track of the order of pages entering memory.

            **Advantages:**
            * Easy to understand and implement.

            **Disadvantages:**
            * Suffers from **Belady's Anomaly** (increasing the number of frames can sometimes *increase* the number of page faults).
            * Does not consider the actual usage frequency of pages, leading to potentially removing frequently used pages.
            """)

        with st.expander("LRU (Least Recently Used) Page Replacement"):
            st.markdown("""
            **Concept:** Replaces the page that has not been used for the longest period of time. This algorithm operates on the principle that pages used recently are more likely to be used again in the near future (locality of reference).

            **How it Works:**
            * Requires tracking the "age" of pages, typically by timestamping their last use or maintaining a linked list where frequently accessed pages move to the front.
            * When a page fault occurs, the page with the oldest timestamp (or at the tail of the linked list) is removed.

            **Advantages:**
            * Generally performs better than FIFO, as it utilizes the principle of locality.
            * Does not suffer from Belady's Anomaly.

            **Disadvantages:**
            * More complex to implement than FIFO, as it requires hardware support or significant overhead to track page usage times.
            * Actual implementation is challenging due to the need for continuous time tracking.
            """)

        with st.expander("LFU (Least Frequently Used) Page Replacement"):
            st.markdown("""
            **Concept:** Replaces the page that has been used the least frequently over a period of time. It assumes that pages used less often in the past are less likely to be used in the future.

            **How it Works:**
            * Maintains a counter for each page in memory, incrementing the counter whenever the page is accessed.
            * When a page fault occurs and memory is full, the page with the lowest frequency count is removed.
            * **Tie-breaking:** If multiple pages have the same minimum frequency, the algorithm typically breaks ties by removing the page that arrived earliest in memory (similar to FIFO for tied pages).

            **Advantages:**
            * Can perform very well if page access patterns are skewed (some pages are used much more than others).

            **Disadvantages:**
            * Can suffer from the problem of a page having a high initial frequency and then never being used again, but remaining in memory.
            * Requires significant overhead to maintain and update frequency counts.
            * Tie-breaking rules are crucial for its practical performance.
            """)

    elif st.session_state.theory_display_mode == "process_scheduling":
        st.info("Displaying theory for **CPU Scheduling Algorithms**.")
        st.subheader("📊 CPU Scheduling Algorithms")
        
        with st.expander("FCFS (First-Come, First-Served) CPU Scheduling"):
            st.markdown("""
            **Concept:** The simplest CPU scheduling algorithm. Processes are executed in the order in which they arrive in the ready queue. It's a non-preemptive algorithm, meaning once a process starts executing, it runs to completion without interruption.

            **How it Works:**
            * Processes are added to a queue upon arrival.
            * The CPU executes the process at the head of the queue until it completes its burst time.

            **Advantages:**
            * Simple to understand and implement.
            * Fair in the sense that processes are served in the order of their arrival.

            **Disadvantages:**
            * Can lead to long waiting times for short processes if a long process arrives first (known as the **Convoy Effect**).
            * Not suitable for time-sharing systems where responsiveness is crucial.
            """)

        with st.expander("SJF (Shortest Job First) CPU Scheduling"):
            st.markdown("""
            **Concept:** This algorithm associates with each process the length of its next CPU burst. When the CPU becomes available, it is assigned to the process that has the smallest next CPU burst. If two processes have the same burst time, FCFS is used to break the tie.

            **How it Works:**
            * **Non-Preemptive SJF:** Once the CPU is assigned to a process, it cannot be preempted until it completes its CPU burst.
            * **Preemptive SJF (Shortest-Remaining-Time-First - SRTF):** If a new process arrives with a CPU burst time less than the remaining time of the currently executing process, the current process is preempted.

            **Advantages:**
            * Gives the minimum average waiting time for a given set of processes.
            * Generally provides very good overall performance.

            **Disadvantages:**
            * **Requires knowing the next CPU burst length:** This is impossible to know in practice, as burst times are unpredictable. It can only be estimated.
            * Can lead to **starvation** of long processes if there's a continuous stream of short processes.
            """)

        with st.expander("Round Robin (RR) CPU Scheduling"):
            st.markdown("""
            **Concept:** A preemptive scheduling algorithm designed for time-sharing systems. It's similar to FCFS but adds preemption to enable switching between processes. Each process is given a small unit of CPU time, called a **time quantum (or time slice)**.

            **How it Works:**
            * Processes are added to a circular queue.
            * The CPU scheduler goes around the circular queue, allocating CPU to each process for a time quantum.
            * If a process completes its burst within the quantum, it releases the CPU.
            * If a process does not complete within the quantum, it is preempted, and its context is saved. It is then added to the tail of the ready queue.

            **Advantages:**
            * Provides fair allocation of CPU time among processes.
            * Good for interactive systems as it provides quick response times (low turnaround time for short jobs).
            * Avoids starvation.

            **Disadvantages:**
            * Performance heavily depends on the size of the time quantum.
                * Too large a quantum: Degenerates to FCFS.
                * Too small a quantum: Leads to frequent context switching, increasing overhead and reducing effective CPU utilization.
            * Higher turnaround time compared to SJF in many cases.
            """)

    else: # theory_display_mode == "all" (default or from Home)
        st.info("Displaying theory for **All Algorithms**.")
        st.subheader("📄 Page Replacement Algorithms")
        
        with st.expander("FIFO (First-In, First-Out) Page Replacement"):
            st.markdown("""
            **Concept:** The simplest page replacement algorithm. It replaces the oldest page in memory, regardless of how recently or how frequently it has been used. It's like a queue where the page that has been in memory the longest is the first one out.

            **How it Works:**
            * When a page fault occurs and memory is full, the page that was loaded earliest into memory is removed.
            * A queue data structure is typically used to keep track of the order of pages entering memory.

            **Advantages:**
            * Easy to understand and implement.

            **Disadvantages:**
            * Suffers from **Belady's Anomaly** (increasing the number of frames can sometimes *increase* the number of page faults).
            * Does not consider the actual usage frequency of pages, leading to potentially removing frequently used pages.
            """)

        with st.expander("LRU (Least Recently Used) Page Replacement"):
            st.markdown("""
            **Concept:** Replaces the page that has not been used for the longest period of time. This algorithm operates on the principle that pages used recently are more likely to be used again in the near future (locality of reference).

            **How it Works:**
            * Requires tracking the "age" of pages, typically by timestamping their last use or maintaining a linked list where frequently accessed pages move to the front.
            * When a page fault occurs, the page with the oldest timestamp (or at the tail of the linked list) is removed.

            **Advantages:**
            * Generally performs better than FIFO, as it utilizes the principle of locality.
            * Does not suffer from Belady's Anomaly.

            **Disadvantages:**
            * More complex to implement than FIFO, as it requires hardware support or significant overhead to track page usage times.
            * Actual implementation is challenging due to the need for continuous time tracking.
            """)

        with st.expander("LFU (Least Frequently Used) Page Replacement"):
            st.markdown("""
            **Concept:** Replaces the page that has been used the least frequently over a period of time. It assumes that pages used less often in the past are less likely to be used in the future.

            **How it Works:**
            * Maintains a counter for each page in memory, incrementing the counter whenever the page is accessed.
            * When a page fault occurs and memory is full, the page with the lowest frequency count is removed.
            * **Tie-breaking:** If multiple pages have the same minimum frequency, the algorithm typically breaks ties by removing the page that arrived earliest in memory (similar to FIFO for tied pages).

            **Advantages:**
            * Can perform very well if page access patterns are skewed (some pages are used much more than others).

            **Disadvantages:**
            * Can suffer from the problem of a page having a high initial frequency and then never being used again, but remaining in memory.
            * Requires significant overhead to maintain and update frequency counts.
            * Tie-breaking rules are crucial for its practical performance.
            """)

        st.subheader("📊 CPU Scheduling Algorithms")
        
        with st.expander("FCFS (First-Come, First-Served) CPU Scheduling"):
            st.markdown("""
            **Concept:** The simplest CPU scheduling algorithm. Processes are executed in the order in which they arrive in the ready queue. It's a non-preemptive algorithm, meaning once a process starts executing, it runs to completion without interruption.

            **How it Works:**
            * Processes are added to a queue upon arrival.
            * The CPU executes the process at the head of the queue until it completes its burst time.

            **Advantages:**
            * Simple to understand and implement.
            * Fair in the sense that processes are served in the order of their arrival.

            **Disadvantages:**
            * Can lead to long waiting times for short processes if a long process arrives first (known as the **Convoy Effect**).
            * Not suitable for time-sharing systems where responsiveness is crucial.
            """)

        with st.expander("SJF (Shortest Job First) CPU Scheduling"):
            st.markdown("""
            **Concept:** This algorithm associates with each process the length of its next CPU burst. When the CPU becomes available, it is assigned to the process that has the smallest next CPU burst. If two processes have the same burst time, FCFS is used to break the tie.

            **How it Works:**
            * **Non-Preemptive SJF:** Once the CPU is assigned to a process, it cannot be preempted until it completes its CPU burst.
            * **Preemptive SJF (Shortest-Remaining-Time-First - SRTF):** If a new process arrives with a CPU burst time less than the remaining time of the currently executing process, the current process is preempted.

            **Advantages:**
            * Gives the minimum average waiting time for a given set of processes.
            * Generally provides very good overall performance.

            **Disadvantages:**
            * **Requires knowing the next CPU burst length:** This is impossible to know in practice, as burst times are unpredictable. It can only be estimated.
            * Can lead to **starvation** of long processes if there's a continuous stream of short processes.
            """)

        with st.expander("Round Robin (RR) CPU Scheduling"):
            st.markdown("""
            **Concept:** A preemptive scheduling algorithm designed for time-sharing systems. It's similar to FCFS but adds preemption to enable switching between processes. Each process is given a small unit of CPU time, called a **time quantum (or time slice)**.

            **How it Works:**
            * Processes are added to a circular queue.
            * The CPU scheduler goes around the circular queue, allocating CPU to each process for a time quantum.
            * If a process completes its burst within the quantum, it releases the CPU.
            * If a process does not complete within the quantum, it is preempted, and its context is saved. It is then added to the tail of the ready queue.

            **Advantages:**
            * Provides fair allocation of CPU time among processes.
            * Good for interactive systems as it provides quick response times (low turnaround time for short jobs).
            * Avoids starvation.

            **Disadvantages:**
            * Performance heavily depends on the size of the time quantum.
                * Too large a quantum: Degenerates to FCFS.
                * Too small a quantum: Leads to frequent context switching, increasing overhead and reducing effective CPU utilization.
            * Higher turnaround time compared to SJF in many cases.
            """)
