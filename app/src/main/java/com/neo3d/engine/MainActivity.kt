package com.neo3d.engine

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
import android.widget.SeekBar
import android.widget.TextView

// Custom Virtual Analog Joystick
class JoystickView(context: Context, private val onMove: (Float, Float) -> Unit) : View(context) {
    private val outerPaint = Paint().apply { color = Color.argb(100, 255, 255, 255); style = Paint.Style.STROKE; strokeWidth = 8f; isAntiAlias = true }
    private val innerPaint = Paint().apply { color = Color.argb(180, 0, 180, 255); style = Paint.Style.FILL; isAntiAlias = true }
    private var centerX = 0f
    private var centerY = 0f
    private var thumbX = 0f
    private var thumbY = 0f
    private var radius = 130f
    private var thumbRadius = 55f

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        centerX = w * 0.5f; centerY = h * 0.5f
        thumbX = centerX; thumbY = centerY
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawCircle(centerX, centerY, radius, outerPaint)
        canvas.drawCircle(thumbX, thumbY, thumbRadius, innerPaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                val dx = event.x - centerX
                val dy = event.y - centerY
                val dist = Math.hypot(dx.toDouble(), dy.toDouble()).toFloat()
                if (dist <= radius) {
                    thumbX = event.x
                    thumbY = event.y
                } else {
                    thumbX = centerX + (dx / dist) * radius
                    thumbY = centerY + (dy / dist) * radius
                }
                val normX = (thumbX - centerX) / radius
                val normY = -(thumbY - centerY) / radius // Inverted Y: Forward is positive
                onMove(normX, normY)
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                thumbX = centerX; thumbY = centerY
                onMove(0f, 0f)
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}

class MainActivity : Activity(), SurfaceHolder.Callback {

    private external fun nativeStart(surface: Surface)
    private external fun nativeResize(width: Int, height: Int)
    private external fun nativeStop()
    private external fun nativeStatus(): String
    private external fun nativeLoadGlb(data: ByteArray): String
    private external fun nativeLook(dx: Float, dy: Float)
    private external fun nativeJoystickMove(inputX: Float, inputY: Float)
    private external fun nativeJump()
    private external fun nativeUpdateSettings(fogDensity: Float, timeOfDay: Float, exposure: Float)
    private external fun nativeResetView()

    private lateinit var status: TextView
    private lateinit var diagnostics: TextView
    private val uiHandler = Handler(Looper.getMainLooper())
    private val diagnosticLines = ArrayDeque<String>()

    private var surfaceReady = false
    private var lastLookX = 0f
    private var lastLookY = 0f
    private var activeLookPointerId = -1

    private var currentFog = 0.045f
    private var currentTime = 14.0f // 2:00 PM
    private var currentExposure = 1.0f

    companion object { init { System.loadLibrary("neo3d") } }

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
        if (::diagnostics.isInitialized) diagnostics.text = diagnosticLines.joinToString("\n")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        )

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF10151D.toInt())
        }

        // --- TOP TOOLBAR ---
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(10, 4, 10, 4)
        }
        status = TextView(this).apply {
            text = "NEO-3D | FPS Physics Engine"
            textSize = 11f
            setTextColor(0xFFE3EAF4.toInt())
            layoutParams = LinearLayout.LayoutParams(0, -2, 1f)
        }

        val settingsBtn = Button(this).apply { text = "SETTINGS"; textSize = 10f }
        val resetBtn = Button(this).apply {
            text = "RESPAWN"
            textSize = 10f
            setOnClickListener { nativeResetView() }
        }
        val importBtn = Button(this).apply {
            text = "IMPORT GLB"
            textSize = 10f
            setOnClickListener {
                val picker = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                    putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("model/gltf-binary", "application/octet-stream"))
                }
                startActivityForResult(picker, 3107)
            }
        }
        bar.addView(status); bar.addView(importBtn); bar.addView(settingsBtn); bar.addView(resetBtn)

        // --- VIEWPORT + HUD OVERLAY ---
        val viewportContainer = FrameLayout(this).apply { layoutParams = LinearLayout.LayoutParams(-1, 0, 1f) }
        val viewport = SurfaceView(this).apply { layoutParams = FrameLayout.LayoutParams(-1, -1) }
        viewport.holder.addCallback(this)

        // Touch Look on Viewport (Aim / Mouselook)
        viewport.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                    val idx = event.actionIndex
                    // Only use touches on the right half for camera look
                    if (event.getX(idx) > viewport.width * 0.4f && activeLookPointerId == -1) {
                        activeLookPointerId = event.getPointerId(idx)
                        lastLookX = event.getX(idx); lastLookY = event.getY(idx)
                    }
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    if (activeLookPointerId != -1) {
                        val ptrIdx = event.findPointerIndex(activeLookPointerId)
                        if (ptrIdx != -1) {
                            val dx = event.getX(ptrIdx) - lastLookX
                            val dy = event.getY(ptrIdx) - lastLookY
                            lastLookX = event.getX(ptrIdx); lastLookY = event.getY(ptrIdx)
                            if (surfaceReady) nativeLook(dx, dy)
                        }
                    }
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP, MotionEvent.ACTION_CANCEL -> {
                    val idx = event.actionIndex
                    if (event.getPointerId(idx) == activeLookPointerId) activeLookPointerId = -1
                    true
                }
                else -> true
            }
        }
        viewportContainer.addView(viewport)

        // Virtual Joystick (Bottom Left)
        val joystick = JoystickView(this) { x, y ->
            if (surfaceReady) nativeJoystickMove(x, y)
        }.apply {
            layoutParams = FrameLayout.LayoutParams(380, 380).apply {
                gravity = Gravity.BOTTOM or Gravity.START
                setMargins(40, 0, 0, 40)
            }
        }
        viewportContainer.addView(joystick)

        // Jump Button (Bottom Right)
        val jumpBtn = Button(this).apply {
            text = "JUMP"
            textSize = 14f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(160, 0, 160, 230))
            layoutParams = FrameLayout.LayoutParams(180, 180).apply {
                gravity = Gravity.BOTTOM or Gravity.END
                setMargins(0, 0, 60, 60)
            }
            setOnClickListener { if (surfaceReady) nativeJump() }
        }
        viewportContainer.addView(jumpBtn)

        // --- FLOATING SETTINGS EDITOR PANEL ---
        val settingsPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(235, 20, 26, 36))
            setPadding(24, 20, 24, 20)
            visibility = View.GONE
            layoutParams = FrameLayout.LayoutParams(600, -2).apply { gravity = Gravity.CENTER }
        }

        fun createSlider(label: String, initial: Int, max: Int, onProgress: (Float) -> Unit): LinearLayout {
            val row = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, 10, 0, 10) }
            val txt = TextView(this).apply { text = label; setTextColor(Color.WHITE); textSize = 11f }
            val bar = SeekBar(this).apply {
                this.max = max
                progress = initial
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(b: SeekBar?, p: Int, fromUser: Boolean) {
                        onProgress(p.toFloat() / max.toFloat())
                        if (surfaceReady) nativeUpdateSettings(currentFog, currentTime, currentExposure)
                    }
                    override fun onStartTrackingTouch(b: SeekBar?) {}
                    override fun onStopTrackingTouch(b: SeekBar?) {}
                })
            }
            row.addView(txt); row.addView(bar)
            return row
        }

        settingsPanel.addView(TextView(this).apply { text = "GRAPHICS & ATMOSPHERE EDITOR"; setTextColor(0xFF78D6FF.toInt()); textSize = 13f })
        settingsPanel.addView(createSlider("Volumetric Fog Density", 45, 100) { currentFog = it * 0.12f })
        settingsPanel.addView(createSlider("Time of Day (Sun Angle)", 60, 100) { currentTime = it * 24.0f })
        settingsPanel.addView(createSlider("ACES Camera Exposure", 50, 100) { currentExposure = 0.5f + it * 1.5f })
        val closeSettings = Button(this).apply {
            text = "CLOSE SETTINGS"
            setOnClickListener { settingsPanel.visibility = View.GONE }
        }
        settingsPanel.addView(closeSettings)
        viewportContainer.addView(settingsPanel)

        settingsBtn.setOnClickListener {
            settingsPanel.visibility = if (settingsPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        // --- DIAGNOSTICS LOG ---
        val logScroll = ScrollView(this).apply {
            setBackgroundColor(0xFF171E28.toInt())
            isFillViewport = true
            addView(TextView(this@MainActivity).also { diagnostics = it; it.setTextColor(0xFFE3EAF4.toInt()); it.textSize = 9f })
        }

        root.addView(bar)
        root.addView(viewportContainer)
        root.addView(logScroll, LinearLayout.LayoutParams(-1, 80))
        setContentView(root)
        uiHandler.post(diagnosticPoll)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != 3107 || resultCode != RESULT_OK || data?.data == null) return
        val bytes = contentResolver.openInputStream(data.data!!)?.readBytes() ?: return
        if (surfaceReady) status.text = nativeLoadGlb(bytes)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        nativeStart(holder.surface); surfaceReady = true
    }
    override fun surfaceChanged(holder: SurfaceHolder, format: Int, w: Int, h: Int) {
        if (surfaceReady) nativeResize(w, h)
    }
    override fun surfaceDestroyed(holder: SurfaceHolder) {
        surfaceReady = false; nativeStop()
    }
}
