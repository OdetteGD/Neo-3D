package com.neo3d.engine

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
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
    private lateinit var status: TextView
    private var surfaceReady = false

    companion object { init { System.loadLibrary("neo3d") } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or View.SYSTEM_UI_FLAG_LAYOUT_STABLE)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(0xFF10151D.toInt()) }
        val bar = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(18, 8, 18, 8) }
        status = TextView(this).apply { text = "NEO-3D  |  Vulkan renderer starting"; textSize = 14f; setTextColor(0xFFE3EAF4.toInt()); layoutParams = LinearLayout.LayoutParams(0, -2, 1f) }
        val refresh = Button(this).apply { text = "Renderer status"; setOnClickListener { status.text = nativeStatus() } }
        bar.addView(status); bar.addView(refresh)
        val viewport = SurfaceView(this)
        viewport.holder.addCallback(this)
        root.addView(bar, LinearLayout.LayoutParams(-1, -2))
        root.addView(viewport, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
    }

    override fun surfaceCreated(holder: SurfaceHolder) { nativeStart(holder.surface); surfaceReady = true; status.text = nativeStatus() }
    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) { if (surfaceReady) nativeResize(width, height) }
    override fun surfaceDestroyed(holder: SurfaceHolder) { surfaceReady = false; nativeStop() }
    override fun onDestroy() { if (surfaceReady) nativeStop(); surfaceReady = false; super.onDestroy() }
}
