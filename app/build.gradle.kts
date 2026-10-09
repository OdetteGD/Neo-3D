plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.neo3d.engine"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.neo3d.engine"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        ndk { 
            abiFilters += listOf("arm64-v8a", "x86_64") 
        }

        externalNativeBuild { 
            cmake { 
                cppFlags += listOf("-std=c++20", "-Wall", "-Wextra")
                // DITO DAPAT NAKALAGAY ANG ARGUMENTS:
                arguments += listOf("-DANDROID_STL=c++_shared")
            } 
        }
    }

    buildTypes { 
        release { 
            isMinifyEnabled = false 
        } 
    }

    externalNativeBuild { 
        cmake { 
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
            // Tinanggal ang arguments dito dahil hindi ito sinusuportahan sa outer block
        } 
    }

    packaging { 
        jniLibs { 
            useLegacyPackaging = true 
        } 
    }

    compileOptions { 
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17 
    }

    kotlinOptions { 
        jvmTarget = "17" 
    }
}
