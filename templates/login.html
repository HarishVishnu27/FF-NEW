<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Login - Zebra Crossing Vehicle Analytics</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #4f46e5, #7e22ce);
            height: 100vh;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .login-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            overflow: hidden;
            position: relative;
        }
        .login-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(90deg, #4f46e5, #7e22ce, #f472b6);
        }
        .form-input {
            transition: all 0.3s ease;
            border: 1px solid #e5e7eb;
        }
        .form-input:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
            transform: translateY(-2px);
        }
        .login-btn {
            background: linear-gradient(90deg, #4f46e5, #7e22ce);
            transition: all 0.3s ease;
        }
        .login-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
        }
        .floating-shape {
            position: absolute;
            border-radius: 50%;
            opacity: 0.2;
            z-index: -1;
            animation: float 10s infinite alternate ease-in-out;
        }
        .shape-1 {
            width: 300px;
            height: 300px;
            background: #4f46e5;
            top: -150px;
            right: -150px;
            animation-duration: 15s;
        }
        .shape-2 {
            width: 200px;
            height: 200px;
            background: #7e22ce;
            bottom: -100px;
            left: -100px;
            animation-duration: 12s;
            animation-delay: 1s;
        }
        .shape-3 {
            width: 150px;
            height: 150px;
            background: #f472b6;
            bottom: 100px;
            right: -50px;
            animation-duration: 8s;
            animation-delay: 0.5s;
        }
        @keyframes float {
            0% { transform: translate(0, 0) rotate(0deg); }
            100% { transform: translate(20px, -20px) rotate(10deg); }
        }
        .error-message {
            animation: shake 0.5s ease-in-out;
        }
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
    </style>
</head>
<body>
    <!-- Floating shapes for background -->
    <div class="floating-shape shape-1"></div>
    <div class="floating-shape shape-2"></div>
    <div class="floating-shape shape-3"></div>

    <div class="max-w-md w-full login-card p-10">
        <div class="text-center mb-8">
            <div class="flex items-center justify-center mb-4">
                <div class="bg-indigo-600 p-3 rounded-full">
                    <i class="fas fa-traffic-light text-4xl text-white"></i>
                </div>
            </div>
            <h1 class="text-3xl font-bold text-gray-800">Zebra Crossing Analytics</h1>
            <p class="text-gray-600 mt-2">Sign in to access the monitoring system</p>
        </div>

        {% if error %}
        <div class="error-message mb-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-md relative">
            <strong class="font-bold"><i class="fas fa-exclamation-circle mr-1"></i> Error!</strong>
            <span class="block sm:inline">{{ error }}</span>
        </div>
        {% endif %}

        <form action="{{ url_for('login') }}" method="post" class="space-y-6">
            <div>
                <label for="username" class="block text-sm font-medium text-gray-700 mb-1">
                    <i class="fas fa-user mr-1 text-indigo-600"></i>
                    Username
                </label>
                <input type="text" id="username" name="username" required
                       class="form-input w-full border rounded-md shadow-sm py-3 px-4 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
            </div>

            <div>
                <label for="password" class="block text-sm font-medium text-gray-700 mb-1">
                    <i class="fas fa-lock mr-1 text-indigo-600"></i>
                    Password
                </label>
                <input type="password" id="password" name="password" required
                       class="form-input w-full border rounded-md shadow-sm py-3 px-4 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
            </div>

            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <input id="remember-me" name="remember-me" type="checkbox"
                           class="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded">
                    <label for="remember-me" class="ml-2 block text-sm text-gray-700">
                        Remember me
                    </label>
                </div>
            </div>

            <div>
                <button type="submit"
                        class="login-btn w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    <i class="fas fa-sign-in-alt mr-2"></i>
                    Sign in
                </button>
            </div>
        </form>

        <div class="mt-8 text-center text-sm text-gray-500">
            <p>Zebra crossing vehicle monitoring system</p>
            <p class="mt-1">© 2025 All rights reserved</p>
            <p class="mt-2 text-indigo-600">Current Date: 2025-04-27</p>
        </div>
    </div>
</body>
</html>
