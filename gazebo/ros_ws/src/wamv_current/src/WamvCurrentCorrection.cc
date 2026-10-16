/*
 * WamvCurrentCorrection: ocean current for the VRX WAM-V.
 *
 * VRX's vrx::SimpleHydrodynamics applies the Fossen-style hull forces
 *     tau_h = -M_A nu_dot + C_A(nu) nu - D(nu) nu
 * using the hull's velocity nu relative to the WORLD, i.e. still water. With an
 * ocean current, the standard marine-vehicle model (Fossen 2011, ch. 10) evaluates
 * the hydrodynamic Coriolis and damping terms on the velocity relative to the
 * WATER, nu_r = nu - nu_c (irrotational current: angular velocity unchanged).
 *
 * This system runs next to SimpleHydrodynamics (VRX is not modified) and adds
 *     tau_corr = [C_A(nu_r) nu_r - D(nu_r) nu_r] - [C_A(nu) nu - D(nu) nu]
 * so the total hull force is exactly the relative-velocity form. The added-mass
 * inertia term is left on nu_dot (current treated as slowly varying).
 *
 * The coefficients must equal the WAM-V's SimpleHydrodynamics ones (defaults below are
 * VRX's wamv_gazebo_dynamics_plugin.xacro). The current is world-frame ENU [m/s],
 * received on a gz-transport topic (gz.msgs.Vector3d), default /vrx/wamv/water_current.
 */
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Eigen>
#include <gz/math/Vector3.hh>
#include <gz/msgs/vector3d.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <sdf/sdf.hh>

namespace stable
{
class WamvCurrentCorrection : public gz::sim::System,
                              public gz::sim::ISystemConfigure,
                              public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm,
                 gz::sim::EventManager &) override
  {
    // Only one instance may act per Gazebo process: re-attaching (e.g. relaunching the bridge
    // without restarting Gazebo) must not double the correction.
    static std::atomic<int> instances{0};
    if (++instances > 1)
    {
      gzwarn << "[WamvCurrentCorrection] already attached; this instance stays inactive" << std::endl;
      return;
    }
    gz::sim::Model model(_entity);
    const auto linkName = _sdf->Get<std::string>("link_name", "wamv/base_link").first;
    this->link = gz::sim::Link(model.LinkByName(_ecm, linkName));
    if (!this->link.Valid(_ecm))
    {
      gzerr << "[WamvCurrentCorrection] link [" << linkName << "] not found" << std::endl;
      return;
    }
    this->link.EnableVelocityChecks(_ecm);
    // VRX wamv_gazebo_dynamics_plugin.xacro values
    this->xDotU = _sdf->Get<double>("xDotU", 0.0).first;
    this->yDotV = _sdf->Get<double>("yDotV", 0.0).first;
    this->xU = _sdf->Get<double>("xU", 100.0).first;
    this->xUU = _sdf->Get<double>("xUU", 150.0).first;
    this->yV = _sdf->Get<double>("yV", 100.0).first;
    this->yVV = _sdf->Get<double>("yVV", 100.0).first;
    const auto topic = _sdf->Get<std::string>("topic", "/vrx/wamv/water_current").first;
    this->node.Subscribe(topic, &WamvCurrentCorrection::OnCurrent, this);
    gzmsg << "[WamvCurrentCorrection] link=" << linkName << " topic=" << topic
          << " xU=" << xU << " xUU=" << xUU << " yV=" << yV << " yVV=" << yVV << std::endl;
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info, gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || !this->link.Valid(_ecm))
      return;
    gz::math::Vector3d currentWorld;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      currentWorld = this->current;
    }
    if (currentWorld == gz::math::Vector3d::Zero)
      return;
    const auto vel = this->link.WorldLinearVelocity(_ecm);
    const auto angVel = this->link.WorldAngularVelocity(_ecm);
    const auto pose = this->link.WorldInertialPose(_ecm);
    if (!vel || !angVel || !pose)
      return;
    const auto rot = pose->Rot();
    const auto v = rot.Inverse() * (*vel);                      // body velocity over ground
    const auto vr = rot.Inverse() * ((*vel) - currentWorld);    // body velocity through water
    const double r = (rot.Inverse() * (*angVel)).Z();

    auto hull = [&](double u, double w) {   // planar C_A(nu) nu - D(nu) nu, surge/sway/yaw
      Eigen::Vector3d f;
      f(0) = this->yDotV * w * r - (this->xU + this->xUU * std::abs(u)) * u;
      f(1) = this->xDotU * u * r - (this->yV + this->yVV * std::abs(w)) * w;
      f(2) = (this->yDotV * w) * u + (this->xDotU * u) * w;
      return f;
    };
    const Eigen::Vector3d corr = hull(vr.X(), vr.Y()) - hull(v.X(), v.Y());
    const gz::math::Vector3d forceWorld = rot.RotateVector(gz::math::Vector3d(corr(0), corr(1), 0.0));
    const gz::math::Vector3d torqueWorld = rot.RotateVector(gz::math::Vector3d(0.0, 0.0, corr(2)));
    this->link.AddWorldWrench(_ecm, forceWorld, torqueWorld);
  }

private:
  void OnCurrent(const gz::msgs::Vector3d &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->current.Set(_msg.x(), _msg.y(), 0.0);   // horizontal current only
  }

  gz::sim::Link link;
  gz::transport::Node node;
  std::mutex mutex;
  gz::math::Vector3d current{0, 0, 0};
  double xDotU{0}, yDotV{0}, xU{100}, xUU{150}, yV{100}, yVV{100};
};
}  // namespace stable

GZ_ADD_PLUGIN(stable::WamvCurrentCorrection, gz::sim::System,
              stable::WamvCurrentCorrection::ISystemConfigure,
              stable::WamvCurrentCorrection::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(stable::WamvCurrentCorrection, "stable::WamvCurrentCorrection")
