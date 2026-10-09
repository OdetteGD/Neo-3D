package com.neo3d.engine

import android.app.Activity
import android.os.Bundle
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
    private external fun nativeOrbit(dx: Float, dy: Float)
    private external fun nativeSetAutoRotate(enabled: Boolean)
    private external fun nativeResetView()
    private lateinit var status: TextView
    private var surfaceReady = false
    private var lastX = 0f
    private var lastY = 0f
    private var autoRotate = true

    companion object { init { System.loadLibrary("neo3d") } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or View.SYSTEM_UI_FLAG_LAYOUT_STABLE)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(0xFF10151D.toInt()) }
        val bar = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(10, 4, 10, 4) }
        status = TextView(this).apply { text = "NEO-3D | Vulkan 3D editor"; textSize = 12f; setTextColor(0xFFE3EAF4.toInt()); layoutParams = LinearLayout.LayoutParams(0, -2, 1f) }
        val rotate = Button(this).apply {
            text = "Auto: ON"
            setOnClickListener { autoRotate = !autoRotate; nativeSetAutoRotate(autoRotate); text = if (autoRotate) "Auto: ON" else "Auto: OFF" }
        }
        val reset = Button(this).apply { text = "Reset view"; setOnClickListener { nativeResetView(); autoRotate = true; rotate.text = "Auto: ON"; status.text = nativeStatus() } }
        val refresh = Button(this).apply { text = "GPU"; setOnClickListener { status.text = nativeStatus() } }
        bar.addView(status); bar.addView(rotate); bar.addView(reset); bar.addView(refresh)
        val viewport = SurfaceView(this)
        viewport.setBackgroundColor(0xFF10151D.toInt())
        viewport.holder.addCallback(this)
        viewport.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> { lastX = event.x; lastY = event.y; true }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.x - lastX
                    val dy = event.y - lastY
                    lastX = event.x; lastY = event.y
                    if (surfaceReady) { nativeSetAutoRotate(false); autoRotate = false; rotate.text = "Auto: OFF"; nativeOrbit(dx, dy) }
                    true
                }
                else -> true
            }
        }
        root.addView(bar, LinearLayout.LayoutParams(-1, -2))
        root.addView(viewport, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
    }

    override fun surfaceCreated(holder: SurfaceHolder) { nativeStart(holder.surface); surfaceReady = true; status.text = nativeStatus() }
    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) { if (surfaceReady) nativeResize(width, height) }
    override fun surfaceDestroyed(holder: SurfaceHolder) { surfaceReady = false; nativeStop() }
    override fun onDestroy() { if (surfaceReady) nativeStop(); surfaceReady = false; super.onDestroy() }
}
