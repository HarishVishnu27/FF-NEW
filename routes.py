from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session
from functools import wraps
import os
import json
from datetime import datetime, timedelta
import csv
from io import StringIO
import numpy as np

from config import (
    PROCESSED_FOLDER, REGIONS, COLOR_MAPPING, REGIONS_FILE,
    DEFAULT_USERNAME, DEFAULT_PASSWORD, FRAME_SKIP
)
import database
import vision_processing

app = Flask(__name__)
# Secret key for session
app.secret_key = 'harish@2704'

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials. Please try again.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    # Dashboard: Show today's stats (from 12 AM local time, not last 24 hours)
    latest_images = {}
    for cam_id in range(1, 5):
        cam_folder = os.path.join(PROCESSED_FOLDER, f'cam{cam_id}')
        if os.path.exists(cam_folder):
            files = [f for f in os.listdir(cam_folder) if f.endswith('.jpg')]
            if files:
                latest_images[f'cam{cam_id}'] = files[-1]
            else:
                latest_images[f'cam{cam_id}'] = None
        else:
            latest_images[f'cam{cam_id}'] = None

    # Get all today's statistics for the dashboard (since 12 AM local)
    camera_stats = {}
    for cam_id in range(1, 5):
        camera_stats[f'cam{cam_id}'] = database.get_today_zebra_crossing_stats(cam_id)

    # Pass IST time for the dashboard
    now_utc = datetime.utcnow()
    now_ist = now_utc + timedelta(hours=5, minutes=30)
    timestamp_ist = now_ist.strftime('%Y-%m-%d %H:%M:%S')

    return render_template('index.html', latest_images=latest_images, camera_stats=camera_stats, timestamp=timestamp_ist)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if request.method == 'POST':
        data = request.json
        cam_id = data.get('cam_id')
        regions = data.get('regions')

        if cam_id and regions:
            REGIONS[f'cam{cam_id}'] = regions
            return jsonify({'status': 'success', 'message': 'Regions updated'}), 200
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    return render_template('admin.html')

@app.route('/update_regions', methods=['POST'])
def update_regions():
    try:
        data = request.json
        cam_id = data.get('cam_id')
        new_regions = data.get('regions')

        if not cam_id or not new_regions:
            return jsonify({'status': 'error', 'message': 'Missing required data'}), 400

        if cam_id not in REGIONS:
            REGIONS[cam_id] = {}

        for region in new_regions:
            region_type = region.get('type')
            vertices = region.get('vertices')
            color = region.get('color', 'blue')

            if not region_type or not vertices or region_type != 'Zebra':
                continue

            REGIONS[cam_id]['Zebra'] = {
                'vertices': np.array(vertices, dtype=np.int32),
                'color': COLOR_MAPPING.get(color.lower(), (0, 0, 255)),
                'weight': 1.0
            }

        with open(REGIONS_FILE, 'w') as file:
            json_regions = {}
            for cam, regions in REGIONS.items():
                json_regions[cam] = {}
                for reg_name, reg_data in regions.items():
                    json_regions[cam][reg_name] = {
                        'vertices': reg_data['vertices'].tolist(),
                        'color': reg_data['color'],
                        'weight': reg_data['weight']
                    }
            json.dump(json_regions, file, indent=4)

        # Clear all processed images for this camera
        vision_processing.clear_camera_images(cam_id)

        return jsonify({
            'status': 'success',
            'message': f'Regions updated successfully for {cam_id} and images cleared'
        }), 200

    except Exception as e:
        print(f"Error updating regions: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error updating regions: {str(e)}'
        }), 500

@app.route('/get_regions', methods=['GET'])
def get_regions():
    try:
        return jsonify(REGIONS), 200
    except Exception as e:
        print(f"Error fetching regions: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_frame/<cam_id>', methods=['GET'])
def get_frame(cam_id):
    try:
        frame = vision_processing.get_single_frame(cam_id)
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to capture frame'}), 500

        filename = f'static/temp/{cam_id}_frame.jpg'
        import cv2
        cv2.imwrite(filename, frame)

        return jsonify({'status': 'success', 'frame_url': f'/{filename}'})
    except Exception as e:
        print(f"Error extracting frame for {cam_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/zebra_crossing_analytics')
@login_required
def zebra_crossing_analytics():
    try:
        # By default, show today's data (from 12 AM local)
        data = {}
        summary = {}
        first_data_times = {}

        midnight_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_today = midnight_today.strftime('%Y-%m-%d %H:%M:%S')
        end_today = (midnight_today + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')

        for cam_id in range(1, 5):
            data[f'cam{cam_id}'] = database.get_zebra_crossing_data_by_date(
                cam_id, start_today, end_today
            )
            summary[f'cam{cam_id}'] = database.get_zebra_crossing_summary(cam_id)
            first_data_times[f'cam{cam_id}'] = summary[f'cam{cam_id}'].get('first_data_time', None)

        return render_template('zebra_crossing.html',
                               data=data, summary=summary, first_data_times=first_data_times)
    except Exception as e:
        print(f"Error in zebra crossing analytics route: {e}")
        return f"Error loading zebra crossing analytics: {str(e)}", 500

@app.route('/api/analytics_data', methods=['GET'])
@login_required
def api_analytics_data():
    """
    API endpoint for AJAX analytics page: returns data for selected camera and filter
    Query params:
      cam_id: camera number (1-4) or "all"
      preset: one of 'today', 'yesterday', 'last24', 'lastweek', 'lastmonth', 'custom'
      start_date, end_date: for custom
      vehicle_type: optional
    """
    try:
        cam_id = request.args.get('cam_id')
        preset = request.args.get('preset', 'today')
        vehicle_type = request.args.get('vehicle_type')
        start_date = None
        end_date = None

        now = datetime.now()
        if preset == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = (start_date + timedelta(days=1))
        elif preset == 'yesterday':
            start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
        elif preset == 'last24':
            end_date = now
            start_date = end_date - timedelta(hours=24)
        elif preset == 'lastweek':
            start_date = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif preset == 'lastmonth':
            start_date = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif preset == 'custom':
            s = request.args.get('start_date')
            e = request.args.get('end_date')
            if s and e:
                start_date = datetime.strptime(s, "%Y-%m-%d")
                end_date = datetime.strptime(e, "%Y-%m-%d") + timedelta(days=1)

        if start_date and end_date:
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            start_str = end_str = None

        # Helper function for a single camera
        def fetch_single_cam(cam):
            if start_str and end_str:
                data = database.get_zebra_crossing_data_by_date(cam, start_str, end_str, vehicle_type=vehicle_type)
                stats = database.get_zebra_crossing_stats_custom_range(cam, start_str, end_str)
            else:
                data = database.get_zebra_crossing_data(cam, vehicle_type=vehicle_type)
                stats = database.get_zebra_crossing_summary(cam)
            return data, stats

        # Handle all cameras combined
        # Handle all cameras combined (accept both "consolidated" and "all")
        if cam_id in ("all", "consolidated", None, ""):

            all_data = []
            combined_stats = {
                'total_count': 0,
                'vehicle_counts': {},
                'since': None,
                'latest_detection': None
            }
            since_list = []
            latest_detection_list = []
            for cam in range(1, 5):
                data, stats = fetch_single_cam(cam)
                all_data.extend(data)
                combined_stats['total_count'] += stats.get('total_count', 0)
                for vt, count in stats.get('vehicle_counts', {}).items():
                    combined_stats['vehicle_counts'][vt] = combined_stats['vehicle_counts'].get(vt, 0) + count
                if stats.get('since'):
                    since_list.append(stats['since'])
                if stats.get('latest_detection'):
                    latest_detection_list.append(stats['latest_detection'])
            # Set combined since as earliest, latest_detection as latest
            combined_stats['since'] = min(since_list) if since_list else None
            combined_stats['latest_detection'] = max(latest_detection_list) if latest_detection_list else None
            return jsonify({'data': all_data, 'stats': combined_stats})
        else:
            cam = int(cam_id)
            data, stats = fetch_single_cam(cam)
            return jsonify({'data': data, 'stats': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_last_processed/<cam_id>')
def get_last_processed(cam_id):
    try:
        processed_folder = os.path.join(PROCESSED_FOLDER, cam_id)
        if not os.path.exists(processed_folder):
            return jsonify({'status': 'error', 'message': 'No processed images found'}), 404

        files = [f for f in os.listdir(processed_folder) if f.endswith('.jpg')]
        if not files:
            return jsonify({'status': 'error', 'message': 'No processed images found'}), 404

        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(processed_folder, x)))
        return jsonify({'status': 'success', 'image_url': f'/static/processed/{cam_id}/{latest_file}'})

    except Exception as e:
        print(f"Error getting last processed image for {cam_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_frame_skip', methods=['POST'])
def update_frame_skip():
    global FRAME_SKIP
    frame_skip = request.json.get('frame_skip')
    try:
        import config
        config.FRAME_SKIP = int(frame_skip)
        return jsonify({'status': 'success', 'message': 'Frame skip count updated successfully'})
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid frame skip value provided'}), 400

@app.route('/download_zebra_data', methods=['GET'])
@login_required
def download_zebra_data():
    import math

    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        cam_id = request.args.get('cam_id')
        preset = request.args.get('preset')

        # --- Date/preset logic ---
        now = datetime.now()
        # Handle preset if present and dates are missing or incomplete
        if (not start_date or not end_date) and preset:
            if preset == 'today':
                start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt = start_dt
            elif preset == 'yesterday':
                start_dt = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt = start_dt
            elif preset == 'last24':
                start_dt = now - timedelta(hours=24)
                end_dt = now
            elif preset == 'lastweek':
                start_dt = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt = now
            elif preset == 'lastmonth':
                start_dt = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt = now
            elif preset == 'custom':
                # If preset is custom but dates are missing, fallback to today
                start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt = start_dt
            else:
                start_dt = now
                end_dt = now
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
        else:
            # If dates are given, always use them
            if not start_date:
                start_date = datetime.now().strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # --- Camera selection logic ---
        if not cam_id or cam_id == "all":
            cam_ids = [1, 2, 3, 4]
        else:
            try:
                cam_ids = [int(cam_id)]
            except Exception:
                cam_ids = [1, 2, 3, 4]

        # --- Date range logic ---
        # For all time buckets, we need integer days from 00:00 of start_dt to 23:59 of end_dt
        date_range_days = (end_dt - start_dt).days + 1
        all_dates = [(start_dt + timedelta(days=i)).strftime('%d-%m-%Y') for i in range(date_range_days)]

        # --- Vehicle type mapping ---
        VEHICLE_COLS = ['car', 'bikes', 'truck', 'bus']
        RAW_TO_BIKE = ['motorcycle', 'bicycle']

        # --- Prepare half-hour slots ---
        time_slots = []
        for h in range(0, 24):
            for m in [0, 30]:
                start = f"{h:02}:{m:02}"
                end_h = h if m == 0 else (h + 1) % 24
                end_m = (m + 30) % 60
                end = f"{end_h:02}:{end_m:02}"
                slot_label = f"{start} to {end}"
                time_slots.append((h, m, slot_label))

        slot_count = len(time_slots)

        # --- Data structures per camera and for all cameras ---
        cam_data = {cam: [{} for _ in range(slot_count * date_range_days)] for cam in cam_ids}
        all_data = [{} for _ in range(slot_count * date_range_days)]

        cam_totals = {cam: dict.fromkeys(VEHICLE_COLS, 0) for cam in cam_ids}
        all_totals = dict.fromkeys(VEHICLE_COLS, 0)
        cam_totals_sum = {cam: 0 for cam in cam_ids}
        all_totals_sum = 0

        # --- Helper to get slot index ---
        def get_slot_index(ts):
            # ts: 'YYYY-MM-DD HH:MM:SS'
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            day_idx = (t.date() - start_dt.date()).days
            slot = t.hour * 2 + (1 if t.minute >= 30 else 0)
            return day_idx * slot_count + slot, t.strftime('%d-%m-%Y')

        # --- For each camera, fetch and bucket data ---
        for cam in cam_ids:
            rows = database.get_zebra_crossing_data_by_date(
                cam,
                start_dt.strftime('%Y-%m-%d'),
                (end_dt + timedelta(days=1)).strftime('%Y-%m-%d'),
                limit=1000000
            )
            for row in rows:
                ts = row['timestamp']
                vtype = row['vehicle_type']
                count = row['vehicle_count']
                idx, date_val = get_slot_index(ts)
                if idx < 0 or idx >= slot_count * date_range_days:
                    continue
                if vtype in RAW_TO_BIKE:
                    vtype_mapped = 'bikes'
                elif vtype == 'car':
                    vtype_mapped = 'car'
                elif vtype == 'truck':
                    vtype_mapped = 'truck'
                elif vtype == 'bus':
                    vtype_mapped = 'bus'
                else:
                    continue
                bucket = cam_data[cam][idx]
                bucket.setdefault('date', date_val)
                bucket.setdefault('slot', time_slots[idx % slot_count][2])
                bucket[vtype_mapped] = bucket.get(vtype_mapped, 0) + count

        # --- Build all cameras combined ---
        for idx in range(slot_count * date_range_days):
            slot_row = {'date': None, 'slot': time_slots[idx % slot_count][2]}
            for cam in cam_ids:
                bucket = cam_data[cam][idx]
                if 'date' in bucket:
                    slot_row['date'] = bucket['date']
                for vcol in VEHICLE_COLS:
                    slot_row[vcol] = slot_row.get(vcol, 0) + bucket.get(vcol, 0)
            if cam_ids and slot_row['date']:
                all_data[idx] = slot_row

        # --- Calculate totals ---
        for cam in cam_ids:
            for idx in range(slot_count * date_range_days):
                bucket = cam_data[cam][idx]
                for vcol in VEHICLE_COLS:
                    cam_totals[cam][vcol] += bucket.get(vcol, 0)
                cam_totals_sum[cam] += sum(bucket.get(vcol, 0) for vcol in VEHICLE_COLS)
        for idx in range(slot_count * date_range_days):
            bucket = all_data[idx]
            for vcol in VEHICLE_COLS:
                all_totals[vcol] += bucket.get(vcol, 0)
            all_totals_sum += sum(bucket.get(vcol, 0) for vcol in VEHICLE_COLS)

        # --- Prepare CSV ---
        si = StringIO()
        cw = csv.writer(si)

        # Header row 1
        header_row1 = ['S. No', 'Date', 'Time Stamp']
        for cam in cam_ids:
            header_row1 += [f'Camera {cam}'] + [''] * 4
        header_row1 += ['Consolidated'] + [''] * 4
        cw.writerow(header_row1)

        # Header row 2
        header_row2 = ['', '', '']
        for _ in cam_ids:
            header_row2 += ['Cars', 'Bikes', 'Trucks', 'Buses', 'Total']
        header_row2 += ['Cars', 'Bikes', 'Trucks', 'Buses', 'Total']
        cw.writerow(header_row2)

        # Data rows
        s_no = 1
        for row_num in range(slot_count * date_range_days):
            # Only write rows if at least one cam or all-cams has data in this slot
            write_row = False
            row = [s_no]
            # Date and slot for this line
            date_val = ''
            for cam in cam_ids:
                if 'date' in cam_data[cam][row_num]:
                    date_val = cam_data[cam][row_num]['date']
                    break
            if not date_val and 'date' in all_data[row_num]:
                date_val = all_data[row_num]['date']
            row.append(date_val or all_dates[0])
            row.append(time_slots[row_num % slot_count][2])

            for cam in cam_ids:
                bucket = cam_data[cam][row_num]
                car = bucket.get('car', 0)
                bikes = bucket.get('bikes', 0)
                truck = bucket.get('truck', 0)
                bus = bucket.get('bus', 0)
                total = car + bikes + truck + bus
                row += [car, bikes, truck, bus, total]
                if total > 0:
                    write_row = True
            # All cameras combined
            bucket = all_data[row_num]
            car = bucket.get('car', 0)
            bikes = bucket.get('bikes', 0)
            truck = bucket.get('truck', 0)
            bus = bucket.get('bus', 0)
            total = car + bikes + truck + bus
            row += [car, bikes, truck, bus, total]
            if total > 0:
                write_row = True
            if write_row:
                cw.writerow(row)
                s_no += 1

        # Totals row
        total_row = ['', 'Total', '']
        for cam in cam_ids:
            t = cam_totals[cam]
            row_total = t['car'] + t['bikes'] + t['truck'] + t['bus']
            total_row += [t['car'], t['bikes'], t['truck'], t['bus'], row_total]
        all_row_total = all_totals['car'] + all_totals['bikes'] + all_totals['truck'] + all_totals['bus']
        total_row += [all_totals['car'], all_totals['bikes'], all_totals['truck'], all_totals['bus'], all_row_total]
        cw.writerow(total_row)

        output = si.getvalue()
        si.close()
        return Response(
            output,
            mimetype="text/csv",
            headers={
                "Content-Disposition":
                f"attachment;filename=zebra_crossing_timeslots_{start_date}_to_{end_date}.csv"
            }
        )
    except Exception as e:
        print(f"Error in download route: {e}")
        return f"Error downloading analytics: {str(e)}", 500
@app.route('/toggle_sahi', methods=['POST'])
@login_required
def toggle_sahi():
    try:
        use_sahi = request.json.get('use_sahi', True)
        use_multi_scale = request.json.get('use_multi_scale', True)

        # Save SAHI configuration to a JSON file for persistence
        sahi_state = {
            'use_sahi': use_sahi,
            'use_multi_scale': use_multi_scale
        }

        with open('sahi_state.json', 'w') as f:
            json.dump(sahi_state, f)

        return jsonify({
            'status': 'success',
            'message': f'SAHI processing {"enabled" if use_sahi else "disabled"}, ' +
                      f'Multi-scale detection {"enabled" if use_multi_scale else "disabled"}'
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add a new route to get and update SAHI configuration
@app.route('/sahi_config', methods=['GET', 'POST'])
@login_required
def sahi_config():
    if request.method == 'POST':
        try:
            data = request.json

            # Update SAHI configuration
            import vision_processing

            # Update confidence threshold
            if 'confidence_threshold' in data:
                vision_processing.SAHI_CONFIG['confidence_threshold'] = float(data['confidence_threshold'])
                # Also update the detection model with new threshold
                vision_processing.detection_model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8',
                    model_path='models/yolov8m.pt',
                    confidence_threshold=vision_processing.SAHI_CONFIG['confidence_threshold'],
                    device=vision_processing.DEVICE
                )

            # Update other parameters
            for key in ['slice_height', 'slice_width']:
                if key in data:
                    vision_processing.SAHI_CONFIG[key] = int(data[key])

            for key in ['overlap_height_ratio', 'overlap_width_ratio', 'postprocess_match_threshold']:
                if key in data:
                    vision_processing.SAHI_CONFIG[key] = float(data[key])

            # Save configuration to a file for persistence
            with open('sahi_config.json', 'w') as f:
                json.dump(vision_processing.SAHI_CONFIG, f)

            return jsonify({
                'status': 'success',
                'message': 'SAHI configuration updated successfully'
            }), 200

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        # Return current SAHI configuration
        import vision_processing
        return jsonify(vision_processing.SAHI_CONFIG), 200
