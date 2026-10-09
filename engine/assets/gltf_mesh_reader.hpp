#pragma once
// Minimal, dependency-free glTF 2.0 mesh extraction for Neo-3D.
// Handles GLB embedded buffers, triangle primitives, POSITION/NORMAL and unsigned indices.
// Unsupported accessor features (sparse accessors, non-triangle modes, external buffers)
// fail explicitly instead of silently producing an empty viewport.
#include "glb_reader.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace neo3d::assets {
namespace detail {
struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object } kind=Kind::Null;
    double number=0;
    bool boolean=false;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string,Json>> object;
    const Json* get(std::string_view key) const {
        if(kind!=Kind::Object) return nullptr;
        for(const auto& p:object) if(p.first==key) return &p.second;
        return nullptr;
    }
    const Json& at(std::size_t i) const {
        if(kind!=Kind::Array || i>=array.size()) throw std::runtime_error("glTF JSON array index out of range");
        return array[i];
    }
    std::size_t size() const { return kind==Kind::Array ? array.size() : 0; }
    std::size_t index() const {
        if(kind!=Kind::Number || number<0 || number>static_cast<double>(std::numeric_limits<std::size_t>::max()) || std::floor(number)!=number)
            throw std::runtime_error("glTF expected a non-negative integer");
        return static_cast<std::size_t>(number);
    }
    double numeric() const { if(kind!=Kind::Number) throw std::runtime_error("glTF expected a number"); return number; }
    std::string text() const { if(kind!=Kind::String) throw std::runtime_error("glTF expected a string"); return string; }
};
class JsonParser {
public:
    explicit JsonParser(std::string_view s):s_(s){}
    Json parse() { skip(); Json v=value(); skip(); if(pos_!=s_.size()) fail("trailing JSON data"); return v; }
private:
    std::string_view s_; std::size_t pos_=0;
    [[noreturn]] void fail(const char* m) const { throw std::runtime_error(std::string("glTF JSON: ")+m); }
    void skip(){while(pos_<s_.size()&&(s_[pos_]==' '||s_[pos_]=='\n'||s_[pos_]=='\r'||s_[pos_]=='\t'))++pos_;}
    bool take(char c){skip();if(pos_<s_.size()&&s_[pos_]==c){++pos_;return true;}return false;}
    std::string str(){
        if(!take('"')) fail("expected string"); std::string out;
        while(pos_<s_.size()){char c=s_[pos_++];if(c=='"')return out;if(static_cast<unsigned char>(c)<0x20)fail("control byte in string");
            if(c!='\\'){out.push_back(c);continue;} if(pos_>=s_.size())fail("truncated escape"); char e=s_[pos_++];
            switch(e){case '"':out.push_back('"');break;case '\\':out.push_back('\\');break;case '/':out.push_back('/');break;case 'b':out.push_back('\b');break;case 'f':out.push_back('\f');break;case 'n':out.push_back('\n');break;case 'r':out.push_back('\r');break;case 't':out.push_back('\t');break;
            case 'u': { if(pos_+4>s_.size())fail("truncated unicode escape"); unsigned cp=0; for(int i=0;i<4;++i){char h=s_[pos_++];cp<<=4;if(h>='0'&&h<='9')cp|=h-'0';else if(h>='a'&&h<='f')cp|=h-'a'+10;else if(h>='A'&&h<='F')cp|=h-'A'+10;else fail("invalid unicode escape");}
                if(cp<0x80)out.push_back(static_cast<char>(cp));else if(cp<0x800){out.push_back(static_cast<char>(0xC0|(cp>>6)));out.push_back(static_cast<char>(0x80|(cp&63)));}else{out.push_back(static_cast<char>(0xE0|(cp>>12)));out.push_back(static_cast<char>(0x80|((cp>>6)&63)));out.push_back(static_cast<char>(0x80|(cp&63)));} break;}
            default:fail("invalid escape");}
        } fail("unterminated string");
    }
    Json value(){
        skip();if(pos_>=s_.size())fail("unexpected end");Json v;char c=s_[pos_];
        if(c=='{'){++pos_;v.kind=Json::Kind::Object;skip();if(take('}'))return v;do{skip();if(pos_>=s_.size()||s_[pos_]!='"')fail("object key missing");auto k=str();if(!take(':'))fail("colon missing");v.object.emplace_back(std::move(k),value());}while(take(','));if(!take('}'))fail("object not closed");return v;}
        if(c=='['){++pos_;v.kind=Json::Kind::Array;skip();if(take(']'))return v;do{v.array.push_back(value());}while(take(','));if(!take(']'))fail("array not closed");return v;}
        if(c=='"'){v.kind=Json::Kind::String;v.string=str();return v;}
        if(s_.substr(pos_,4)=="true"){pos_+=4;v.kind=Json::Kind::Bool;v.boolean=true;return v;}
        if(s_.substr(pos_,5)=="false"){pos_+=5;v.kind=Json::Kind::Bool;return v;}
        if(s_.substr(pos_,4)=="null"){pos_+=4;return v;}
        std::size_t begin=pos_;if(c=='-')++pos_;while(pos_<s_.size()&&s_[pos_]>='0'&&s_[pos_]<='9')++pos_;if(pos_<s_.size()&&s_[pos_]=='.'){++pos_;while(pos_<s_.size()&&s_[pos_]>='0'&&s_[pos_]<='9')++pos_;}if(pos_<s_.size()&&(s_[pos_]=='e'||s_[pos_]=='E')){++pos_;if(pos_<s_.size()&&(s_[pos_]=='+'||s_[pos_]=='-'))++pos_;while(pos_<s_.size()&&s_[pos_]>='0'&&s_[pos_]<='9')++pos_;}
        if(begin==pos_)fail("invalid value");try{v.number=std::stod(std::string(s_.substr(begin,pos_-begin)));}catch(...){fail("invalid number");}v.kind=Json::Kind::Number;return v;
    }
};
inline const Json& required(const Json& j,const char* key){auto p=j.get(key);if(!p)throw std::runtime_error(std::string("glTF missing ")+key);return *p;}
inline std::size_t optionalIndex(const Json& j,const char* key,std::size_t fallback){auto p=j.get(key);return p?p->index():fallback;}
inline std::uint32_t u32(const std::vector<std::uint8_t>& b,std::size_t p){if(p>b.size()||b.size()-p<4)throw std::runtime_error("glTF read exceeds buffer");return std::uint32_t(b[p])|(std::uint32_t(b[p+1])<<8)|(std::uint32_t(b[p+2])<<16)|(std::uint32_t(b[p+3])<<24);}
}

struct GltfVertex { float position[3]{0,0,0}; float normal[3]{0,1,0}; };
struct GltfPrimitive { std::vector<GltfVertex> vertices; std::vector<std::uint32_t> indices; std::size_t material=static_cast<std::size_t>(-1); float baseColorFactor[4]{1.0f,1.0f,1.0f,1.0f}; float metallicFactor=1.0f; float roughnessFactor=1.0f; };
struct GltfMeshDocument { std::vector<GltfPrimitive> primitives; std::size_t skippedPrimitives=0; };

inline GltfMeshDocument readGlbMeshes(const std::vector<std::uint8_t>& bytes) {
    using detail::Json;
    const auto container=readGlb(bytes);
    const Json root=detail::JsonParser(container.json).parse();
    const auto& buffers=detail::required(root,"buffers");
    if(buffers.size()>1) throw std::runtime_error("GLB importer currently supports one embedded buffer");
    if(buffers.size()==0) throw std::runtime_error("glTF has no buffer");
    const auto& buffer0=buffers.at(0);
    const auto expected=detail::required(buffer0,"byteLength").index();
    if(expected>container.binary.size()) throw std::runtime_error("glTF buffer byteLength exceeds GLB BIN chunk");
    const auto& views=detail::required(root,"bufferViews");
    const auto& accessors=detail::required(root,"accessors");
    auto viewRange=[&](std::size_t accessorIndex,std::size_t componentBytes,std::size_t components,std::size_t& offset,std::size_t& stride,std::size_t& count,std::size_t& componentType){
        const auto& a=accessors.at(accessorIndex);
        if(a.get("sparse")) throw std::runtime_error("Sparse glTF accessors are not supported yet");
        count=detail::required(a,"count").index();componentType=detail::required(a,"componentType").index();
        const auto type=detail::required(a,"type").text();
        const std::size_t actualComponents=type=="SCALAR"?1:type=="VEC2"?2:type=="VEC3"?3:type=="VEC4"?4:0;
        if(actualComponents!=components) throw std::runtime_error("glTF accessor has an unexpected element type");
        const auto vi=detail::required(a,"bufferView").index();const auto& bv=views.at(vi);
        if(detail::optionalIndex(bv,"buffer",0)!=0)throw std::runtime_error("External glTF buffers are not supported by GLB importer");
        offset=detail::optionalIndex(bv,"byteOffset",0)+detail::optionalIndex(a,"byteOffset",0);
        const auto viewLength=detail::required(bv,"byteLength").index();
        const auto viewStart=detail::optionalIndex(bv,"byteOffset",0);
        stride=detail::optionalIndex(bv,"byteStride",componentBytes*components);
        if(stride<componentBytes*components)throw std::runtime_error("glTF byteStride is smaller than element size");
        if(offset<viewStart || offset-viewStart>viewLength)throw std::runtime_error("glTF accessor starts outside bufferView");
        if(count && (count-1)>(std::numeric_limits<std::size_t>::max()-(componentBytes*components))/stride)throw std::runtime_error("glTF accessor size overflow");
        const std::size_t end=offset+(count?((count-1)*stride+componentBytes*components):0);
        if(end>viewStart+viewLength||end>expected)throw std::runtime_error("glTF accessor exceeds bufferView or buffer");
    };
    // Read core glTF metallic-roughness material factors (textures are handled separately).
    struct MaterialFactors { float base[4]{1,1,1,1}; float metallic=1, roughness=1; };
    std::vector<MaterialFactors> materials;
    if (const auto* list=root.get("materials")) {
        materials.resize(list->size());
        for (std::size_t mi=0; mi<list->size(); ++mi) {
            const auto& material=list->at(mi);
            const auto* pbr=material.get("pbrMetallicRoughness");
            if (!pbr) continue;
            auto readFactor=[&](const Json* value, float* dst, std::size_t n, const char* name) {
                if (!value) return;
                if (value->kind!=Json::Kind::Array || value->size()!=n)
                    throw std::runtime_error(std::string("glTF material ")+name+" must be an array of "+std::to_string(n)+" numbers");
                for (std::size_t k=0;k<n;++k) {
                    const double v=value->at(k).numeric();
                    if (!std::isfinite(v)) throw std::runtime_error(std::string("glTF material ")+name+" contains non-finite value");
                    dst[k]=static_cast<float>(v);
                }
            };
            readFactor(pbr->get("baseColorFactor"),materials[mi].base,4,"baseColorFactor");
            if (const auto* v=pbr->get("metallicFactor")) materials[mi].metallic=static_cast<float>(v->numeric());
            if (const auto* v=pbr->get("roughnessFactor")) materials[mi].roughness=static_cast<float>(v->numeric());
            materials[mi].metallic=std::clamp(materials[mi].metallic,0.0f,1.0f);
            materials[mi].roughness=std::clamp(materials[mi].roughness,0.0f,1.0f);
            for (float& v : materials[mi].base) v=std::clamp(v,0.0f,1.0f);
        }
    }
    const auto& meshes=detail::required(root,"meshes");GltfMeshDocument out;
    for(std::size_t mi=0;mi<meshes.size();++mi){
        const auto& mesh=meshes.at(mi);const auto& prims=detail::required(mesh,"primitives");
        for(std::size_t pi=0;pi<prims.size();++pi){
            const auto& p=prims.at(pi);const auto mode=detail::optionalIndex(p,"mode",4);
            if(mode!=4){++out.skippedPrimitives;continue;}
            const auto* attrs=p.get("attributes");if(!attrs||!attrs->get("POSITION"))throw std::runtime_error("glTF triangle primitive has no POSITION attribute");
            const auto posIndex=attrs->get("POSITION")->index();std::size_t posOff=0,posStride=0,posCount=0,posType=0;
            viewRange(posIndex,4,3,posOff,posStride,posCount,posType);if(posType!=5126)throw std::runtime_error("glTF POSITION must use FLOAT components");
            const auto& posAcc=accessors.at(posIndex);const auto posView=detail::required(posAcc,"bufferView").index();const auto& posBv=views.at(posView);
            if(detail::optionalIndex(posBv,"buffer",0)!=0)throw std::runtime_error("External glTF buffer not supported");
            GltfPrimitive result;result.vertices.resize(posCount);
            for(std::size_t i=0;i<posCount;++i)for(std::size_t c=0;c<3;++c){float f;std::memcpy(&f,container.binary.data()+posOff+i*posStride+c*4,4);if(!std::isfinite(f))throw std::runtime_error("glTF POSITION contains non-finite value");result.vertices[i].position[c]=f;}
            if(const auto* ni=attrs->get("NORMAL")){
                std::size_t no=0,ns=0,nc=0,nt=0;viewRange(ni->index(),4,3,no,ns,nc,nt);
                if(nt!=5126||nc!=posCount)throw std::runtime_error("glTF NORMAL accessor must be FLOAT VEC3 and match POSITION count");
                for(std::size_t i=0;i<nc;++i){float n[3];std::memcpy(n,container.binary.data()+no+i*ns,sizeof(n));const float len=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);if(len>1e-8f)for(int c=0;c<3;++c)result.vertices[i].normal[c]=n[c]/len;}
            }
            if(const auto* ix=p.get("indices")){
                const auto& ia=accessors.at(ix->index());const auto type=detail::required(ia,"componentType").index();
                const std::size_t bytesPer=type==5121?1:type==5123?2:type==5125?4:0;
                if(!bytesPer)throw std::runtime_error("glTF indices must be unsigned byte, short, or int");
                std::size_t io=0,is=0,ic=0,it=0;viewRange(ix->index(),bytesPer,1,io,is,ic,it);
                if(is<bytesPer)throw std::runtime_error("glTF index stride is smaller than component size");
                result.indices.reserve(ic);
                for(std::size_t i=0;i<ic;++i){const auto at=io+i*is;std::uint32_t value=0;if(type==5121)value=container.binary.at(at);else if(type==5123)value=std::uint32_t(container.binary.at(at))|(std::uint32_t(container.binary.at(at+1))<<8);else value=detail::u32(container.binary,at);if(value>=posCount)throw std::runtime_error("glTF index exceeds POSITION accessor");result.indices.push_back(value);}
            } else {result.indices.resize(posCount);for(std::size_t i=0;i<posCount;++i)result.indices[i]=static_cast<std::uint32_t>(i);}
            if(result.indices.size()%3!=0)throw std::runtime_error("glTF triangle index count is not divisible by three");
            if(const auto* mat=p.get("material")) {
                result.material=mat->index();
                if (result.material>=materials.size()) throw std::runtime_error("glTF primitive references a missing material");
                const auto& factors=materials[result.material];
                std::copy(factors.base,factors.base+4,result.baseColorFactor);
                result.metallicFactor=factors.metallic;
                result.roughnessFactor=factors.roughness;
            }
            out.primitives.push_back(std::move(result));
        }
    }
    if(out.primitives.empty())throw std::runtime_error("GLB contains no supported triangle mesh primitives");
    return out;
}
} // namespace neo3d::assets
