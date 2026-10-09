package com.neo3d.engine

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.MotionEvent
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity(), SurfaceHolder.Callback {

    // Native JNI functions na konektado sa C++ Vulkan Renderer
    private external fun nativeStart(surface: Surface)
    private external fun nativeResize(width: Int, height: Int)
    private external fun nativeStop()
    private external fun nativeStatus(): String
    private external fun nativeLoadGlb(data: ByteArray): String
    private external fun nativeOrbit(dx: Float, dy: Float)
    private external fun nativeZoom(delta: Float)
    private external fun nativeMoveCamera(forward: Float, right: Float)
    private external fun nativeElevateCamera(up: Float)
    private external fun nativeSetAutoRotate(enabled: Boolean)
    private external fun nativeResetView()

    private lateinit var status: TextView
    private lateinit var diagnostics: TextView
    private val uiHandler = Handler(Looper.getMainLooper())
    private val diagnosticLines = ArrayDeque<String>()

    private var surfaceReady = false
    private var lastX = 0f
    private var lastY = 0f
    private var pinchStartDist = 0f
    private var autoRotate = true
    private val importRequestCode = 3107

    companion object {
        init {
            System.loadLibrary("neo3d")
        }
    }

    private val diagnosticPoll = object : Runnable {
        override fun run() {
            if (surfaceReady) {
                val current = nativeStatus()
                status.text = current
                if (diagnosticLines.lastOrNull() != current) addDiagnostic(current)
            }
            uiHandler.postDelayed(this, 1000)
        }
    }

    private fun addDiagnostic(message: String) {
        val stamp = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())
        diagnosticLines.addLast("[$stamp] $message")
        while (diagnosticLines.size > 8) diagnosticLines.removeFirst()
        if (::diagnostics.isInitialized) {
            diagnostics.text = diagnosticLines.joinToString("\n")
        }
        android.util.Log.i("Neo3D-Diagnostics", message)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Immersive Fullscreen Mode
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_FULLSCREEN or
            View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        )

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF10151D.toInt())
        }

        // ==========================================
        // 1. TOP TOOLBAR
        // ==========================================
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(10, 4, 10, 4)
        }

        status = TextView(this).apply {
            text = "NEO-3D | Vulkan 3D editor"
            textSize = 11f
            setTextColor(0xFFE3EAF4.toInt())
            layoutParams = LinearLayout.LayoutParams(0, -2, 1f)
        }

        val rotateBtn = Button(this).apply {
            text = "AUTO: ON"
            textSize = 11f
            setOnClickListener {
                autoRotate = !autoRotate
                nativeSetAutoRotate(autoRotate)
                text = if (autoRotate) "AUTO: ON" else "AUTO: OFF"
            }
        }

        val resetBtn = Button(this).apply {
            text = "RESET VIEW"
            textSize = 11f
            setOnClickListener {
                nativeResetView()
                autoRotate = true
                rotateBtn.text = "AUTO: ON"
                status.text = nativeStatus()
            }
        }

        val gpuBtn = Button(this).apply {
            text = "GPU"
            textSize = 11f
            setOnClickListener {
                status.text = nativeStatus()
            }
        }

        val importBtn = Button(this).apply {
            text = "IMPORT GLB"
            textSize = 11f
            setOnClickListener {
                val picker = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                    putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("model/gltf-binary", "application/octet-stream"))
                }
                startActivityForResult(picker, importRequestCode)
            }
        }

        bar.addView(status)
        bar.addView(importBtn)
        bar.addView(rotateBtn)
        bar.addView(resetBtn)
        bar.addView(gpuBtn)

        // ==========================================
        // 2. VIEWPORT + D-PAD OVERLAY CONTAINER
        // ==========================================
        val viewportContainer = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(-1, 0, 1f)
        }

        // Vulkan SurfaceView (Strictly NO background color!)
        val viewport = SurfaceView(this).apply {
            layoutParams = FrameLayout.LayoutParams(-1, -1)
        }
        viewport.holder.addCallback(this)

        // Touch gestures: 1-finger Orbit, 2-finger Pinch Zoom
        viewport.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    lastX = event.x
                    lastY = event.y
                    true
                }
                MotionEvent.ACTION_POINTER_DOWN -> {
                    if (event.pointerCount == 2) {
                        val dx = event.getX(0) - event.getX(1)
                        val dy = event.getY(0) - event.getY(1)
                        pinchStartDist = Math.hypot(dx.toDouble(), dy.toDouble()).toFloat()
                    }
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    if (event.pointerCount == 1) {
                        val dx = event.x - lastX
                        val dy = event.y - lastY
                        lastX = event.x
                        lastY = event.y
                        if (surfaceReady) {
                            nativeOrbit(dx, dy)
                        }
                    } else if (event.pointerCount >= 2 && pinchStartDist > 10f) {
                        val dx = event.getX(0) - event.getX(1)
                        val dy = event.getY(0) - event.getY(1)
                        val currentDist = Math.hypot(dx.toDouble(), dy.toDouble()).toFloat()
                        val delta = (currentDist - pinchStartDist) * 0.015f
                        if (surfaceReady) {
                            nativeZoom(delta)
                        }
                        pinchStartDist = currentDist
                    }
                    true
                }
                else -> true
            }
        }
        viewportContainer.addView(viewport)

        // ==========================================
        // 3. 3D CAMERA D-PAD CONTROLLER (OVERLAY)
        // ==========================================
        val dpad = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(16, 16, 16, 16)
            setBackgroundColor(0x77000000.toInt()) // Semi-transparent black background
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.BOTTOM or Gravity.START
                setMargins(20, 0, 0, 20)
            }
        }

        val btnUp = Button(this).apply {
            text = "▲"
            textSize = 14f
            layoutParams = LinearLayout.LayoutParams(110, 85).apply { gravity = Gravity.CENTER_HORIZONTAL }
        }

        val dpadHorizontal = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }

        val btnLeft = Button(this).apply {
            text = "◀"
            textSize = 14f
            layoutParams = LinearLayout.LayoutParams(110, 85)
        }

        val btnDown = Button(this).apply {
            text = "▼"
            textSize = 14f
            layoutParams = LinearLayout.LayoutParams(110, 85)
        }

        val btnRight = Button(this).apply {
            text = "▶"
            textSize = 14f
            layoutParams = LinearLayout.LayoutParams(110, 85)
        }

        dpadHorizontal.addView(btnLeft)
        dpadHorizontal.addView(btnDown)
        dpadHorizontal.addView(btnRight)

        dpad.addView(btnUp)
        dpad.addView(dpadHorizontal)

        // I-attach ang smooth hold-to-move para tuloy-tuloy ang lakad ng 3D camera
        setupHoldToMove(btnUp)    { if (surfaceReady) nativeMoveCamera(1.0f, 0.0f) }   // Forward
        setupHoldToMove(btnDown)  { if (surfaceReady) nativeMoveCamera(-1.0f, 0.0f) }  // Backward
        setupHoldToMove(btnLeft)  { if (surfaceReady) nativeMoveCamera(0.0f, -1.0f) }  // Strafe Left
        setupHoldToMove(btnRight) { if (surfaceReady) nativeMoveCamera(0.0f, 1.0f) }   // Strafe Right

        viewportContainer.addView(dpad)

        // ==========================================
        // 4. DIAGNOSTICS LOG VIEW
        // ==========================================
        val logTitle = TextView(this).apply {
            text = "RENDER DIAGNOSTICS · latest 8 events"
            textSize = 10f
            setTextColor(0xFF78D6FF.toInt())
            setPadding(8, 4, 8, 2)
        }

        diagnostics = TextView(this).apply {
            text = "Waiting for Vulkan surface initialization…"
            textSize = 10f
            typeface = android.graphics.Typeface.MONOSPACE
            setTextColor(0xFFE3EAF4.toInt())
            setPadding(8, 2, 8, 6)
        }

        val logScroll = ScrollView(this).apply {
            setBackgroundColor(0xFF171E28.toInt())
            isFillViewport = true
            addView(diagnostics)
        }

        root.addView(bar, LinearLayout.LayoutParams(-1, -2))
        root.addView(viewportContainer)
        root.addView(logTitle, LinearLayout.LayoutParams(-1, -2))
        root.addView(logScroll, LinearLayout.LayoutParams(-1, 95))

        setContentView(root)
        addDiagnostic("UI started; awaiting Vulkan SurfaceView")
        uiHandler.post(diagnosticPoll)
    }

    // Helper: Gumagalaw nang tuloy-tuloy habang nakadiin ang daliri sa D-Pad button
    private fun setupHoldToMove(button: Button, action: () -> Unit) {
        val moveHandler = Handler(Looper.getMainLooper())
        val moveRunnable = object : Runnable {
            override fun run() {
                action()
                moveHandler.postDelayed(this, 30) // ~33 FPS continuous camera step
            }
        }

        button.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    action()
                    moveHandler.postDelayed(moveRunnable, 100)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    moveHandler.removeCallbacks(moveRunnable)
                    true
                }
                else -> false
            }
        }
    }

    @Deprecated("Uses document picker result callback for broad Android compatibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != importRequestCode || resultCode != RESULT_OK || data?.data == null) return
        try {
            val bytes = contentResolver.openInputStream(data.data!!)?.use { input ->
                val output = java.io.ByteArrayOutputStream()
                val buffer = ByteArray(64 * 1024)
                var total = 0
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    total += count
                    if (total > 128 * 1024 * 1024) throw IllegalArgumentException("GLB exceeds 128 MiB limit")
                    output.write(buffer, 0, count)
                }
                output.toByteArray()
            } ?: throw IllegalStateException("Unable to open selected file")

            if (!surfaceReady) throw IllegalStateException("Wait for the 3D viewport to start before importing")
            status.text = nativeLoadGlb(bytes)
            addDiagnostic(status.text.toString())
        } catch (e: Exception) {
            val err = "Import failed: ${e.message ?: "unable to read file"}"
            status.text = err
            addDiagnostic(err)
        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        addDiagnostic("Surface created; starting native Vulkan renderer")
        nativeStart(holder.surface)
        surfaceReady = true
        status.text = nativeStatus()
        addDiagnostic(status.text.toString())
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        if (surfaceReady) {
            addDiagnostic("Surface changed: ${width}x${height}")
            nativeResize(width, height)
        }
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        addDiagnostic("Surface destroyed; stopping Vulkan renderer")
        surfaceReady = false
        nativeStop()
    }

    override fun onDestroy() {
        uiHandler.removeCallbacks(diagnosticPoll)
        if (surfaceReady) nativeStop()
        surfaceReady = false
        super.onDestroy()
    }
}
