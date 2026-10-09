package com.neo3d.engine

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.ScrollView
import android.view.Gravity
import android.view.MotionEvent
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity(), SurfaceHolder.Callback {
    private external fun nativeStart(surface: Surface)
    private external fun nativeResize(width: Int, height: Int)
    private external fun nativeStop()
    private external fun nativeStatus(): String
    private external fun nativeLoadGlb(data: ByteArray): String
    private external fun nativeOrbit(dx: Float, dy: Float)
    private external fun nativeSetAutoRotate(enabled: Boolean)
    private external fun nativeResetView()
    private external fun nativeZoom(delta: Float)
    private external fun nativeMoveCamera(forward: Float, right: Float)

    private lateinit var status: TextView
    private lateinit var diagnostics: TextView
    private val uiHandler = Handler(Looper.getMainLooper())
    private val diagnosticLines = ArrayDeque<String>()

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
        // Inayos mula sa "\\n" patungong "\n" para maayos ang multiline display
        if (::diagnostics.isInitialized) diagnostics.text = diagnosticLines.joinToString("\n")
        android.util.Log.i("Neo3D-Diagnostics", message)
    }

    private var surfaceReady = false
    private var lastX = 0f
    private var lastY = 0f
    private var lastPinchDistance = 0f
    private var joystickForward = 0f
    private var joystickRight = 0f
    private val joystickHandler = Handler(Looper.getMainLooper())
    private val joystickTick = object : Runnable {
        override fun run() {
            if (surfaceReady && (joystickForward != 0f || joystickRight != 0f)) {
                nativeMoveCamera(joystickForward, joystickRight)
                joystickHandler.postDelayed(this, 32)
            }
        }
    }
    private var autoRotate = true
    private val importRequestCode = 3107

    companion object {
        init {
            System.loadLibrary("neo3d")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

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

        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(10, 4, 10, 4)
        }

        status = TextView(this).apply {
            text = "NEO-3D | Vulkan 3D editor"
            textSize = 12f
            setTextColor(0xFFE3EAF4.toInt())
            layoutParams = LinearLayout.LayoutParams(0, -2, 1f)
        }

        val rotate = Button(this).apply {
            text = "Auto: ON"
            setOnClickListener {
                autoRotate = !autoRotate
                nativeSetAutoRotate(autoRotate)
                text = if (autoRotate) "Auto: ON" else "Auto: OFF"
            }
        }

        val reset = Button(this).apply {
            text = "Reset view"
            setOnClickListener {
                nativeResetView()
                autoRotate = true
                rotate.text = "Auto: ON"
                status.text = nativeStatus()
            }
        }

        val refresh = Button(this).apply {
            text = "GPU"
            setOnClickListener {
                status.text = nativeStatus()
            }
        }

        val import = Button(this).apply {
            text = "Import GLB"
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
        bar.addView(import)
        bar.addView(rotate)
        bar.addView(reset)
        bar.addView(refresh)

        // --- FIXED: SurfaceView setup ---
        val viewport = SurfaceView(this)
        // HUWAG maglagay ng viewport.setBackgroundColor dito para hindi matakpan ang Vulkan!
        viewport.holder.addCallback(this)

        viewport.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    lastX = event.x
                    lastY = event.y
                    lastPinchDistance = 0f
                    true
                }
                MotionEvent.ACTION_POINTER_DOWN -> {
                    if (event.pointerCount >= 2) lastPinchDistance = pointerDistance(event)
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    if (event.pointerCount >= 2) {
                        val distance = pointerDistance(event)
                        if (lastPinchDistance > 0f && distance > 0f && surfaceReady) {
                            nativeZoom((lastPinchDistance - distance) * 0.018f)
                        }
                        lastPinchDistance = distance
                    } else {
                        val dx = event.x - lastX
                        val dy = event.y - lastY
                        if (surfaceReady) {
                            nativeSetAutoRotate(false)
                            autoRotate = false
                            rotate.text = "Auto: OFF"
                            nativeOrbit(dx, dy)
                        }
                    }
                    lastX = event.x
                    lastY = event.y
                    true
                }
                MotionEvent.ACTION_POINTER_UP -> {
                    lastPinchDistance = 0f
                    true
                }
                else -> true
            }
        }

        root.addView(bar, LinearLayout.LayoutParams(-1, -2))
        val viewportFrame = android.widget.FrameLayout(this)
        viewportFrame.addView(viewport, android.widget.FrameLayout.LayoutParams(-1, -1))
        val joystick = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(4, 4, 4, 4)
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(0x66303848)
                cornerRadius = 18f
            }
        }
        fun joystickButton(label: String, forward: Float, right: Float) = Button(this).apply {
            text = label
            textSize = 12f
            minWidth = 44
            setPadding(2, 0, 2, 0)
            setOnTouchListener { _, ev ->
                when (ev.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        joystickForward = forward
                        joystickRight = right
                        joystickHandler.removeCallbacks(joystickTick)
                        joystickHandler.post(joystickTick)
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        joystickForward = 0f
                        joystickRight = 0f
                        joystickHandler.removeCallbacks(joystickTick)
                    }
                }
                true
            }
        }
        joystick.addView(joystickButton("▲", 1f, 0f))
        val joystickRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(joystickButton("◀", 0f, -1f))
            addView(joystickButton("▼", -1f, 0f))
            addView(joystickButton("▶", 0f, 1f))
        }
        joystick.addView(joystickRow)
        val joystickParams = android.widget.FrameLayout.LayoutParams(-2, -2, android.view.Gravity.BOTTOM or android.view.Gravity.START)
        joystickParams.setMargins(12, 12, 12, 12)
        viewportFrame.addView(joystick, joystickParams)
        root.addView(viewportFrame, LinearLayout.LayoutParams(-1, 0, 1f))

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

        root.addView(logTitle, LinearLayout.LayoutParams(-1, -2))
        root.addView(logScroll, LinearLayout.LayoutParams(-1, 104))

        setContentView(root)
        addDiagnostic("UI started; awaiting Vulkan SurfaceView")
        uiHandler.post(diagnosticPoll)
    }

    private fun pointerDistance(event: MotionEvent): Float {
        if (event.pointerCount < 2) return 0f
        val dx = event.getX(0) - event.getX(1)
        val dy = event.getY(0) - event.getY(1)
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    @Deprecated("Uses the platform document picker result callback for broad Android compatibility")
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
        } catch (e: Exception) {
            status.text = "Import failed: ${e.message ?: "unable to read file"}"
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
        joystickHandler.removeCallbacks(joystickTick)
        joystickForward = 0f
        joystickRight = 0f
        if (surfaceReady) nativeStop()
        surfaceReady = false
        super.onDestroy()
    }
}
