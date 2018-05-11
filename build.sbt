name := "simple-neural-network"

version := "0.1"

scalaVersion := "2.12.6"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "1.0-RC2"
  , "org.scalanlp" %% "breeze-natives" % "1.0-RC2"
  , "org.scalanlp" %% "breeze-viz" % "1.0-RC2"
  , "org.platanios" %% "tensorflow" % "0.1.1" classifier "linux-cpu-x86_64"
  , "org.platanios" %% "tensorflow-api" % "0.1.1"
  , "org.platanios" %% "tensorflow-data" % "0.1.1"
  , "org.platanios" %% "tensorflow-jni" % "0.1.1"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
  , "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
  , Resolver.sonatypeRepo("snapshots")
)