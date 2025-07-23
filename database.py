import sqlite3
from datetime import datetime, timedelta
from config import DATABASE

def init_db(database_name=DATABASE):
    """Initialize database with required tables"""
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    for cam_id in range(1, 5):
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS zebra_crossing_data_cam{cam_id} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                x_position INTEGER NOT NULL,
                y_position INTEGER NOT NULL,
                road_side TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        ''')

        # Create indexes for faster queries
        cursor.execute(f'''
            CREATE INDEX IF NOT EXISTS idx_timestamp_cam{cam_id} 
            ON zebra_crossing_data_cam{cam_id}(timestamp)
        ''')

        cursor.execute(f'''
            CREATE INDEX IF NOT EXISTS idx_vehicle_type_cam{cam_id}
            ON zebra_crossing_data_cam{cam_id}(vehicle_type)
        ''')

    conn.commit()
    conn.close()

def save_zebra_crossing_data(cam_id, vehicle_type, x_position, y_position, road_side, confidence):
    """Save zebra crossing detection data"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute(f'''INSERT INTO zebra_crossing_data_cam{cam_id}
                           (timestamp, vehicle_type, x_position, y_position, road_side, confidence)
                           VALUES (?, ?, ?, ?, ?, ?)''',
                       (timestamp, vehicle_type, x_position, y_position, road_side, confidence))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Database error for camera {cam_id}: {e}")
        return False

def get_first_data_timestamp(cam_id):
    """Get the timestamp of the first data record for a camera"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        query = f'''
            SELECT MIN(timestamp) FROM zebra_crossing_data_cam{cam_id}
        '''
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except sqlite3.Error as e:
        print(f"Error fetching first timestamp for camera {cam_id}: {e}")
        return None

def get_zebra_crossing_data(cam_id, limit=500, start_date=None, end_date=None, vehicle_type=None):
    """Get zebra crossing data for a camera, optionally filtered by date and/or vehicle type"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        query = f'''
            SELECT 
                timestamp, vehicle_type, 
                COUNT(*) as vehicle_count
            FROM zebra_crossing_data_cam{cam_id}
        '''
        conditions = []
        params = []

        if start_date and end_date:
            conditions.append("timestamp BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        elif start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        elif end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if vehicle_type:
            conditions.append("vehicle_type = ?")
            params.append(vehicle_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += '''
            GROUP BY timestamp, vehicle_type
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()

        processed_rows = [
            {
                'timestamp': row[0],
                'vehicle_type': row[1],
                'vehicle_count': row[2]
            } for row in rows
        ]

        conn.close()
        return processed_rows
    except sqlite3.Error as e:
        print(f"Error fetching zebra crossing data for camera {cam_id}: {e}")
        return []

def get_zebra_crossing_data_by_date(cam_id, start_date, end_date, limit=1000, vehicle_type=None):
    """Get zebra crossing data for a camera within a date range, optionally filter by vehicle type"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        query = f'''
            SELECT 
                timestamp, vehicle_type, 
                COUNT(*) as vehicle_count
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp BETWEEN ? AND ?
        '''
        params = [start_date, end_date]

        if vehicle_type:
            query += " AND vehicle_type = ?"
            params.append(vehicle_type)

        query += '''
            GROUP BY timestamp, vehicle_type
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()

        processed_rows = [
            {
                'timestamp': row[0],
                'vehicle_type': row[1],
                'vehicle_count': row[2]
            } for row in rows
        ]

        conn.close()
        return processed_rows
    except sqlite3.Error as e:
        print(f"Error fetching zebra crossing data for camera {cam_id}: {e}")
        return []

def get_today_zebra_crossing_stats(cam_id):
    """Get today's (from 12 AM local time) stats for zebra crossing data"""
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Calculate midnight today (IST)
        now = datetime.now()
        midnight_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = midnight_today.strftime('%Y-%m-%d %H:%M:%S')

        # Get total count by vehicle type since midnight
        query = f'''
            SELECT 
                vehicle_type, 
                COUNT(*) as count
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp >= ?
            GROUP BY vehicle_type
            ORDER BY count DESC
        '''
        cursor.execute(query, (cutoff_time,))
        vehicle_counts = {row['vehicle_type']: row['count'] for row in cursor.fetchall()}

        # Get total count
        query = f'''
            SELECT 
                COUNT(*) as total_count
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp >= ?
        '''
        cursor.execute(query, (cutoff_time,))
        total_count = cursor.fetchone()['total_count']

        # Get latest detection time
        query = f'''
            SELECT 
                MAX(timestamp) as latest_detection
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp >= ?
        '''
        cursor.execute(query, (cutoff_time,))
        latest_detection = cursor.fetchone()['latest_detection']

        conn.close()

        return {
            'total_count': total_count,
            'vehicle_counts': vehicle_counts,
            'latest_detection': latest_detection,
            'since': cutoff_time
        }
    except sqlite3.Error as e:
        print(f"Error fetching today zebra crossing stats for camera {cam_id}: {e}")
        return {
            'total_count': 0,
            'vehicle_counts': {},
            'latest_detection': None,
            'since': None
        }

def get_recent_zebra_crossing_stats(cam_id, hours=1):
    """Get statistics for zebra crossing data in the last few hours (for preset filters)"""
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row  # Return results as dictionaries
        cursor = conn.cursor()

        # Calculate cutoff time
        cutoff_time = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')

        # Get total count by vehicle type
        query = f'''
            SELECT 
                vehicle_type, 
                COUNT(*) as count
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp > ?
            GROUP BY vehicle_type
            ORDER BY count DESC
        '''
        cursor.execute(query, (cutoff_time,))
        vehicle_counts = {row['vehicle_type']: row['count'] for row in cursor.fetchall()}

        # Get total count
        query = f'''
            SELECT 
                COUNT(*) as total_count
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp > ?
        '''
        cursor.execute(query, (cutoff_time,))
        total_count = cursor.fetchone()['total_count']

        # Get latest detection time
        query = f'''
            SELECT 
                MAX(timestamp) as latest_detection
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp > ?
        '''
        cursor.execute(query, (cutoff_time,))
        latest_detection = cursor.fetchone()['latest_detection']

        conn.close()

        return {
            'total_count': total_count,
            'vehicle_counts': vehicle_counts,
            'latest_detection': latest_detection,
            'since': cutoff_time
        }
    except sqlite3.Error as e:
        print(f"Error fetching recent zebra crossing stats for camera {cam_id}: {e}")
        return {
            'total_count': 0,
            'vehicle_counts': {},
            'latest_detection': None,
            'since': None
        }

def get_zebra_crossing_stats_custom_range(cam_id, start_date, end_date):
    """Get zebra crossing stats for a custom date range (for preset filters)"""
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get total count by vehicle type
        query = f'''
            SELECT 
                vehicle_type, 
                COUNT(*) as count
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY vehicle_type
            ORDER BY count DESC
        '''
        cursor.execute(query, (start_date, end_date))
        vehicle_counts = {row['vehicle_type']: row['count'] for row in cursor.fetchall()}

        # Get total count
        query = f'''
            SELECT 
                COUNT(*) as total_count
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp BETWEEN ? AND ?
        '''
        cursor.execute(query, (start_date, end_date))
        total_count = cursor.fetchone()['total_count']

        # Get latest detection time
        query = f'''
            SELECT 
                MAX(timestamp) as latest_detection
            FROM zebra_crossing_data_cam{cam_id}
            WHERE timestamp BETWEEN ? AND ?
        '''
        cursor.execute(query, (start_date, end_date))
        latest_detection = cursor.fetchone()['latest_detection']

        conn.close()

        return {
            'total_count': total_count,
            'vehicle_counts': vehicle_counts,
            'latest_detection': latest_detection,
            'since': start_date
        }
    except sqlite3.Error as e:
        print(f"Error fetching zebra crossing stats for camera {cam_id}: {e}")
        return {
            'total_count': 0,
            'vehicle_counts': {},
            'latest_detection': None,
            'since': start_date
        }

def get_zebra_crossing_summary(cam_id):
    """Get summary statistics for zebra crossing data"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Get total vehicles by type
        query = f'''
            SELECT 
                vehicle_type, 
                COUNT(*) as count
            FROM zebra_crossing_data_cam{cam_id}
            GROUP BY vehicle_type
            ORDER BY count DESC
        '''
        cursor.execute(query)
        vehicle_type_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Get total vehicles
        query = f'''
            SELECT 
                COUNT(*) as total
            FROM zebra_crossing_data_cam{cam_id}
        '''
        cursor.execute(query)
        total_count = cursor.fetchone()[0]

        # Get vehicles by hour of day (for peak hour analysis)
        query = f'''
            SELECT 
                SUBSTR(timestamp, 12, 2) as hour, 
                COUNT(*) as count
            FROM zebra_crossing_data_cam{cam_id}
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
        '''
        cursor.execute(query)
        peak_hour_row = cursor.fetchone()
        peak_hour = peak_hour_row[0] if peak_hour_row else "00"

        # Convert to 12-hour format with AM/PM
        peak_hour_int = int(peak_hour) if peak_hour else 0
        peak_hour_12h = f"{peak_hour_int if peak_hour_int < 13 else peak_hour_int - 12} {'AM' if peak_hour_int < 12 else 'PM'}"

        # Get the first data timestamp
        query = f'''
            SELECT MIN(timestamp) FROM zebra_crossing_data_cam{cam_id}
        '''
        cursor.execute(query)
        first_data_time = cursor.fetchone()[0]

        conn.close()

        # Find most common vehicle type
        most_common_type = max(vehicle_type_counts.items(), key=lambda x: x[1])[0] if vehicle_type_counts else "None"

        return {
            'total_count': total_count,
            'vehicle_types': vehicle_type_counts,
            'most_common_type': most_common_type,
            'peak_hour': peak_hour_12h,
            'first_data_time': first_data_time
        }
    except sqlite3.Error as e:
        print(f"Error fetching zebra crossing summary for camera {cam_id}: {e}")
        return {
            'total_count': 0,
            'vehicle_types': {},
            'most_common_type': "None",
            'peak_hour': "N/A",
            'first_data_time': None
        }
